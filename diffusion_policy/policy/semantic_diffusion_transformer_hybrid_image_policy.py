from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import diffusers.optimization as diffusers_optimization

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_semantic_diffusion import TransformerForSemanticDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmoc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


def _install_robomimic_diffusers_compat():
    """Expose typing symbols removed from modern diffusers for RoboCasa Robomimic."""
    aliases = {
        "Union": Union,
        "Optional": Optional,
        "Optimizer": torch.optim.Optimizer,
    }
    for name, value in aliases.items():
        if not hasattr(diffusers_optimization, name):
            setattr(diffusers_optimization, name, value)


class SemanticDiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            vision_backbone="resnet18",
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            num_subtask_classes=1,
            num_semantic_skill_classes=1,
            num_stage_classes=1,
            semantic_enabled=False,
            direct_boundary_enabled=False,
            bifurcation_enabled=False,
            train_diffusion=True,
            freeze_transformer=False,
            lambda_sem=0.1,
            lambda_bound=0.1,
            direct_subtask_loss_weight=0.25,
            direct_stage_loss_weight=0.25,
            probing_timesteps=(80, 70, 60, 50, 40, 30, 20),
            m_train=4,
            m_eval=16,
            bifurcation_temperature=0.1,
            probe_batch_size_per_device=2,
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw
        
        if vision_backbone == "resnet18":
            config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18Conv"
        elif vision_backbone == "resnet34":
            config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet34Conv"
        elif vision_backbone == "resnet50":
            config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet50Conv"

        if "lang_emb" in obs_config['low_dim']:
            config.observation.encoder.rgb.core_class = "VisualCoreLanguageConditioned"
            if vision_backbone == "resnet18":
                config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet18ConvFiLM"
            elif vision_backbone == "resnet34":
                config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet34ConvFiLM"
            elif vision_backbone == "resnet50":
                config.observation.encoder.rgb.core_kwargs.backbone_class = "ResNet50ConvFiLM"
            else:
                raise NotImplementedError
        
        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        _install_robomimic_diffusers_compat()
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmoc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForSemanticDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        self.semantic_enabled = bool(semantic_enabled)
        self.direct_boundary_enabled = bool(direct_boundary_enabled)
        self.bifurcation_enabled = bool(bifurcation_enabled)
        self.train_diffusion = bool(train_diffusion)
        self.freeze_transformer = bool(freeze_transformer)
        self.lambda_sem = float(lambda_sem)
        self.lambda_bound = float(lambda_bound)
        self.direct_subtask_loss_weight = float(direct_subtask_loss_weight)
        self.direct_stage_loss_weight = float(direct_stage_loss_weight)
        self.probing_timesteps = tuple(int(x) for x in probing_timesteps)
        self.m_train = int(m_train)
        self.m_eval = int(m_eval)
        self.bifurcation_temperature = float(bifurcation_temperature)
        self.probe_batch_size_per_device = int(probe_batch_size_per_device)

        self.semantic_pool_score = nn.Linear(n_emb, 1)
        self.semantic_heads = nn.ModuleDict({
            "subtask": nn.Linear(n_emb, int(num_subtask_classes)),
            "semantic_skill": nn.Linear(n_emb, int(num_semantic_skill_classes)),
            "stage": nn.Linear(n_emb, int(num_stage_classes)),
        })
        self.direct_boundary_heads = nn.ModuleDict({
            name: nn.Linear(n_emb, 1)
            for name in (
                "subtask_future_h8", "stage_future_h8", "any_future_h8"
            )
        })
        self.bifurcation_calibrator = nn.Linear(1, 1)
        self.register_buffer("subtask_class_weight", torch.ones(int(num_subtask_classes)))
        self.register_buffer("stage_class_weight", torch.ones(int(num_stage_classes)))
        self.register_buffer("boundary_pos_weight", torch.ones(3))
        self.obs_encoder.requires_grad_(False)
        self.model.requires_grad_(not self.freeze_transformer)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_auxiliary_loss_weights(self, subtask_class_weight=None,
            stage_class_weight=None, boundary_pos_weight=None):
        for name, value in (
            ("subtask_class_weight", subtask_class_weight),
            ("stage_class_weight", stage_class_weight),
        ):
            if value is None:
                continue
            target = getattr(self, name)
            value = torch.as_tensor(value, dtype=target.dtype, device=target.device)
            if value.shape != target.shape:
                raise ValueError(f"{name}: {value.shape} != {target.shape}")
            target.copy_(value)
        if boundary_pos_weight is not None:
            value = torch.as_tensor(
                boundary_pos_weight,
                dtype=self.boundary_pos_weight.dtype,
                device=self.boundary_pos_weight.device,
            )
            if value.shape != self.boundary_pos_weight.shape:
                raise ValueError(
                    f"boundary_pos_weight: {value.shape} "
                    f"!= {self.boundary_pos_weight.shape}"
                )
            self.boundary_pos_weight.copy_(value)

    def _attention_pool(self, hidden):
        weight = torch.softmax(self.semantic_pool_score(hidden), dim=1)
        return torch.sum(weight * hidden, dim=1)

    def _semantic_logits(self, hidden):
        pooled = self._attention_pool(hidden)
        return {name: head(pooled) for name, head in self.semantic_heads.items()}

    def _direct_boundary_logits(self, hidden):
        pooled = self._attention_pool(hidden)
        return {
            name: head(pooled).squeeze(-1)
            for name, head in self.direct_boundary_heads.items()
        }

    def get_optimizer(self, transformer_weight_decay, obs_encoder_weight_decay,
            learning_rate=None, base_learning_rate=None, aux_learning_rate=None,
            betas=(0.9, 0.95)):
        base_lr = base_learning_rate or learning_rate or 1e-4
        aux_lr = aux_learning_rate or base_lr
        groups = []
        if not self.freeze_transformer:
            for group in self.model.get_optim_groups(transformer_weight_decay):
                params = [p for p in group["params"] if p.requires_grad]
                if params:
                    groups.append({
                        "params": params,
                        "weight_decay": group["weight_decay"],
                        "lr": base_lr,
                    })
        aux_params = []
        for module in (
            self.semantic_pool_score, self.semantic_heads,
            self.direct_boundary_heads, self.bifurcation_calibrator,
        ):
            aux_params.extend(p for p in module.parameters() if p.requires_grad)
        if aux_params:
            groups.append({
                "params": aux_params,
                "weight_decay": transformer_weight_decay,
                "lr": aux_lr,
            })
        if not groups:
            raise ValueError("no trainable parameters")
        return torch.optim.AdamW(groups, betas=betas)

    @staticmethod
    def _masked_ce(logits, target, valid, weight):
        if not torch.any(valid):
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], target[valid], weight=weight)

    @staticmethod
    def _binary_metrics(logits, target):
        prediction = logits >= 0
        truth = target >= 0.5
        true_positive = (prediction & truth).float().sum()
        false_positive = (prediction & ~truth).float().sum()
        false_negative = (~prediction & truth).float().sum()
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = torch.where(
            denominator > 0,
            2 * true_positive / denominator,
            denominator,
        )
        accuracy = (prediction == truth).float().mean()
        return accuracy, f1

    def _training_inputs(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        action = self.normalizer["action"].normalize(batch["action"])
        B, To = action.shape[0], self.n_obs_steps
        if not self.obs_as_cond:
            raise NotImplementedError("semantic policy requires obs_as_cond=True")
        flat_obs = dict_apply(
            nobs, lambda x: x[:, :To].reshape(-1, *x.shape[2:])
        )
        with torch.no_grad():
            features = self.obs_encoder(flat_obs)
        cond = features.reshape(B, To, -1)
        trajectory = action
        if self.pred_action_steps_only:
            trajectory = action[:, To - 1:To - 1 + self.n_action_steps]
        mask = (
            torch.zeros_like(trajectory, dtype=torch.bool)
            if self.pred_action_steps_only else self.mask_generator(trajectory.shape)
        )
        noise = torch.randn_like(trajectory)
        timestep = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,),
            device=trajectory.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timestep)
        noisy[mask] = trajectory[mask]
        return trajectory, noise, timestep, noisy, mask, cond

    def compute_loss(self, batch, auxiliary_scale=1.0):
        trajectory, noise, timestep, noisy, mask, cond = self._training_inputs(batch)
        need_hidden = self.semantic_enabled or self.direct_boundary_enabled
        if need_hidden:
            pred, hidden = self.model(noisy, timestep, cond, return_hidden=True)
        else:
            pred, hidden = self.model(noisy, timestep, cond), None

        kind = self.noise_scheduler.config.prediction_type
        target = noise if kind == "epsilon" else trajectory
        loss_diff = F.mse_loss(pred, target, reduction="none")
        loss_diff = reduce(
            loss_diff * (~mask).to(loss_diff.dtype), "b ... -> b (...)", "mean"
        ).mean()
        total = loss_diff if self.train_diffusion else loss_diff.detach() * 0
        zero = loss_diff.detach() * 0
        result = {
            "loss_diff": loss_diff,
            "loss_sem": zero,
            "loss_subtask": zero,
            "loss_stage": zero,
            "loss_bound": zero,
            "semantic_accuracy_subtask": zero,
            "semantic_accuracy_stage": zero,
            "boundary_accuracy": zero,
            "boundary_subtask_accuracy": zero,
            "boundary_stage_accuracy": zero,
            "boundary_any_f1": zero,
            "boundary_subtask_f1": zero,
            "boundary_stage_f1": zero,
            "boundary_inconsistency_rate": zero,
            "loss_bound_any": zero,
            "loss_bound_subtask": zero,
            "loss_bound_stage": zero,
        }

        if self.semantic_enabled:
            label = batch["semantic_target"]
            valid = label["valid"].bool()
            logits = self._semantic_logits(hidden)
            sub = self._masked_ce(
                logits["subtask"], label["subtask"].long(), valid,
                self.subtask_class_weight,
            )
            stage = self._masked_ce(
                logits["stage"], label["stage"].long(), valid,
                self.stage_class_weight,
            )
            sem = 0.5 * sub + 0.5 * stage
            total = total + auxiliary_scale * self.lambda_sem * sem
            result.update(loss_sem=sem, loss_subtask=sub, loss_stage=stage)
            if torch.any(valid):
                result["semantic_accuracy_subtask"] = (
                    logits["subtask"][valid].argmax(-1) == label["subtask"][valid]
                ).float().mean()
                result["semantic_accuracy_stage"] = (
                    logits["stage"][valid].argmax(-1) == label["stage"][valid]
                ).float().mean()

        if self.direct_boundary_enabled:
            label = batch["boundary_target"]
            valid = label["valid"].bool()
            logits_by_field = self._direct_boundary_logits(hidden)
            field_specs = (
                ("subtask_future_h8", self.direct_subtask_loss_weight, 0),
                ("stage_future_h8", self.direct_stage_loss_weight, 1),
                ("any_future_h8", 1.0, 2),
            )
            bound = logits_by_field["any_future_h8"].sum() * 0
            if torch.any(valid):
                for field, field_weight, weight_index in field_specs:
                    field_logits = logits_by_field[field][valid]
                    field_target = label[field][valid].to(field_logits.dtype)
                    field_loss = F.binary_cross_entropy_with_logits(
                        field_logits,
                        field_target,
                        pos_weight=self.boundary_pos_weight[weight_index],
                    )
                    bound = bound + field_weight * field_loss
                    short_name = field.removesuffix("_future_h8")
                    result[f"loss_bound_{short_name}"] = field_loss
                    accuracy, f1 = self._binary_metrics(
                        field_logits, field_target
                    )
                    metric_name = (
                        "boundary_accuracy"
                        if short_name == "any"
                        else f"boundary_{short_name}_accuracy"
                    )
                    result[metric_name] = accuracy
                    result[f"boundary_{short_name}_f1"] = f1
                probability = {
                    field: torch.sigmoid(logits_by_field[field][valid])
                    for field, _, _ in field_specs
                }
                component_max = torch.maximum(
                    probability["subtask_future_h8"],
                    probability["stage_future_h8"],
                )
                result["boundary_inconsistency_rate"] = (
                    probability["any_future_h8"] + 1e-6 < component_max
                ).float().mean()
            total = total + auxiliary_scale * self.lambda_bound * bound
            result["loss_bound"] = bound
        result["loss_total"] = total
        return result

    @staticmethod
    def _entropy(probability):
        probability = probability.clamp_min(1e-8)
        return -(probability * probability.log()).sum(-1)

    def _js(self, logits):
        probability = torch.softmax(logits, dim=-1)
        return (
            self._entropy(probability.mean(1))
            - self._entropy(probability).mean(1)
        )

    def _encode_condition(self, obs):
        normalized = self.normalizer.normalize(obs)
        B, To = next(iter(normalized.values())).shape[0], self.n_obs_steps
        flat = dict_apply(
            normalized, lambda x: x[:, :To].reshape(-1, *x.shape[2:])
        )
        with torch.no_grad():
            feature = self.obs_encoder(flat)
        return feature.reshape(B, To, -1)

    def _reverse_probes(self, cond, count):
        B = cond.shape[0]
        repeated = cond.repeat_interleave(count, dim=0)
        trajectory = torch.randn(
            B * count, self.horizon, self.action_dim,
            dtype=cond.dtype, device=cond.device,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        requested, probes = set(self.probing_timesteps), {}
        with torch.no_grad():
            for timestep in self.noise_scheduler.timesteps:
                value = int(timestep.item())
                if value in requested:
                    probes[value] = trajectory.detach().clone()
                output = self.model(trajectory, timestep, repeated)
                trajectory = self.noise_scheduler.step(
                    output, timestep, trajectory, **self.kwargs
                ).prev_sample
        missing = requested - set(probes)
        if missing:
            raise ValueError(f"scheduler omitted probing timesteps: {sorted(missing)}")
        return probes

    def compute_bifurcation_loss(self, batch, auxiliary_scale=1.0):
        if not self.bifurcation_enabled:
            raise RuntimeError("bifurcation branch disabled")
        B = min(
            self.probe_batch_size_per_device,
            next(iter(batch["obs"].values())).shape[0],
        )
        obs = {key: value[:B] for key, value in batch["obs"].items()}
        label = {key: value[:B] for key, value in batch["boundary_target"].items()}
        cond = self._encode_condition(obs)
        was_training = self.model.training
        self.model.eval()
        probes = self._reverse_probes(cond, self.m_train)
        repeated = cond.repeat_interleave(self.m_train, dim=0)
        scores = []
        for value in self.probing_timesteps:
            _, hidden = self.model(
                probes[value], torch.tensor(value, device=cond.device), repeated,
                return_hidden=True,
            )
            logits = self._semantic_logits(hidden)
            sub = logits["subtask"].reshape(B, self.m_train, -1)
            stage = logits["stage"].reshape(B, self.m_train, -1)
            scores.append(0.5 * (self._js(sub) + self._js(stage)))
        score = torch.stack(scores, -1)
        temp = self.bifurcation_temperature
        bifurcation = temp * (
            torch.logsumexp(score / temp, -1)
            - math.log(len(self.probing_timesteps))
        )
        logits = self.bifurcation_calibrator(bifurcation[:, None]).squeeze(-1)
        valid = label["valid"].bool()
        if torch.any(valid):
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], label["any_future_h8"][valid].to(logits.dtype),
                pos_weight=self.boundary_pos_weight[2],
            )
            accuracy = (
                (logits[valid] >= 0)
                == (label["any_future_h8"][valid] >= 0.5)
            ).float().mean()
        else:
            loss, accuracy = logits.sum() * 0, logits.detach().sum() * 0
        self.model.train(was_training)
        return {
            "loss_bifurcation": auxiliary_scale * self.lambda_bound * loss,
            "loss_bound_b": loss,
            "bifurcation_score_mean": bifurcation.mean(),
            "boundary_b_accuracy": accuracy,
        }

    @torch.no_grad()
    def predict_bifurcation(self, obs_dict, sample_count=None):
        """Return calibrated boundary probability and per-timestep disagreement."""
        if not self.bifurcation_enabled:
            raise RuntimeError("bifurcation branch disabled")
        count = self.m_eval if sample_count is None else int(sample_count)
        if count < 2:
            raise ValueError("sample_count must be at least 2")
        cond = self._encode_condition(obs_dict)
        batch_size = cond.shape[0]
        was_training = self.model.training
        self.model.eval()
        probes = self._reverse_probes(cond, count)
        repeated = cond.repeat_interleave(count, dim=0)
        scores = []
        for value in self.probing_timesteps:
            _, hidden = self.model(
                probes[value],
                torch.tensor(value, device=cond.device),
                repeated,
                return_hidden=True,
            )
            logits = self._semantic_logits(hidden)
            subtask = logits["subtask"].reshape(batch_size, count, -1)
            stage = logits["stage"].reshape(batch_size, count, -1)
            scores.append(0.5 * (self._js(subtask) + self._js(stage)))
        per_timestep = torch.stack(scores, dim=-1)
        temperature = self.bifurcation_temperature
        score = temperature * (
            torch.logsumexp(per_timestep / temperature, dim=-1)
            - math.log(len(self.probing_timesteps))
        )
        logit = self.bifurcation_calibrator(score[:, None]).squeeze(-1)
        self.model.train(was_training)
        return {
            "probability": torch.sigmoid(logit),
            "logit": logit,
            "score": score,
            "per_timestep_score": per_timestep,
            "sample_count": count,
        }

    def forward(self, batch, auxiliary_scale=1.0, run_bifurcation=False):
        losses = self.compute_loss(batch, auxiliary_scale=auxiliary_scale)
        if run_bifurcation:
            bifurcation = self.compute_bifurcation_loss(
                batch, auxiliary_scale=auxiliary_scale
            )
            losses["loss_total"] = (
                losses["loss_total"] + bifurcation["loss_bifurcation"]
            )
            losses.update(bifurcation)
        return losses
