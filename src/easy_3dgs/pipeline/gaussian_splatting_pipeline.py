import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import field
from typing_extensions import Literal

# Import necessary components from simple_trainer
from easy_3dgs.pipeline.gaussian_splatting.simple_trainer import Config, main as simple_trainer_main, DefaultStrategy, MCMCStrategy

class GaussianSplattingPipeline:
    """Orchestrates the Gaussian Splatting training process."""

    def __init__(
        self,
        # Parameters directly mapping to simple_trainer.Config
        data_dir: Optional[str] = None,
        data_factor: int = 4,
        result_dir: str = "./results/",
        max_steps: int = 30_000,
        eval_steps: Optional[List[int]] = None,
        save_steps: Optional[List[int]] = None,
        save_ply: bool = True,
        ply_steps: Optional[List[int]] = None,
        disable_video: bool = False,
        init_type: str = "sfm",
        sh_degree: int = 3,
        sh_degree_interval: int = 1000,
        init_opa: float = 0.1,
        init_scale: float = 1.0,
        ssim_lambda: float = 0.2,
        antialiased: bool = False,
        random_bkgd: bool = False,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        opacities_lr: float = 5e-2,
        quats_lr: float = 1e-3,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3 / 20,
        opacity_reg: float = 0.0,
        scale_reg: float = 0.0,
        pose_opt: bool = False,
        pose_opt_lr: float = 1e-5,
        pose_opt_reg: float = 1e-6,
        pose_noise: float = 0.0,
        app_opt: bool = False,
        app_embed_dim: int = 16,
        app_opt_lr: float = 1e-3,
        app_opt_reg: float = 1e-6,
        use_bilateral_grid: bool = False,
        bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8),
        depth_loss: bool = False,
        depth_lambda: float = 1e-2,
        tb_every: int = 100,
        tb_save_image: bool = False,
        lpips_net: Literal["vgg", "alex"] = "alex",
        with_ut: bool = False,
        with_eval3d: bool = False,
        use_fused_bilagrid: bool = False,
        steps_scaler: float = 1.0,
        # Strategy selection
        strategy_type: Literal["default", "mcmc"] = "default",
        # Other parameters
        disable_viewer: bool = False,
        port: int = 8080,
        batch_size: int = 1,
        # For distributed training, if needed in the future
        local_rank: int = 0,
        world_rank: int = 0,
        world_size: int = 1,
    ):
        self.config_params = {
            "data_factor": data_factor,
            "result_dir": result_dir,
            "max_steps": max_steps,
            "eval_steps": eval_steps if eval_steps is not None else [-1],
            "save_steps": save_steps if save_steps is not None else [7_000, 30_000],
            "save_ply": save_ply,
            "ply_steps": ply_steps if ply_steps is not None else [30_000],
            "disable_video": disable_video,
            "init_type": init_type,
            "sh_degree": sh_degree,
            "sh_degree_interval": sh_degree_interval,
            "init_opa": init_opa,
            "init_scale": init_scale,
            "ssim_lambda": ssim_lambda,
            "antialiased": antialiased,
            "random_bkgd": random_bkgd,
            "means_lr": means_lr,
            "scales_lr": scales_lr,
            "opacities_lr": opacities_lr,
            "quats_lr": quats_lr,
            "sh0_lr": sh0_lr,
            "shN_lr": shN_lr,
            "opacity_reg": opacity_reg,
            "scale_reg": scale_reg,
            "pose_opt": pose_opt,
            "pose_opt_lr": pose_opt_lr,
            "pose_opt_reg": pose_opt_reg,
            "pose_noise": pose_noise,
            "app_opt": app_opt,
            "app_embed_dim": app_embed_dim,
            "app_opt_lr": app_opt_lr,
            "app_opt_reg": app_opt_reg,
            "use_bilateral_grid": use_bilateral_grid,
            "bilateral_grid_shape": bilateral_grid_shape,
            "depth_loss": depth_loss,
            "depth_lambda": depth_lambda,
            "tb_every": tb_every,
            "tb_save_image": tb_save_image,
            "lpips_net": lpips_net,
            "with_ut": with_ut,
            "with_eval3d": with_eval3d,
            "use_fused_bilagrid": use_fused_bilagrid,
            "steps_scaler": steps_scaler,
            "disable_viewer": disable_viewer,
            "port": port,
            "batch_size": batch_size,
        }

        if strategy_type == "default":
            self.config_params["strategy"] = DefaultStrategy(verbose=True)
        elif strategy_type == "mcmc":
            self.config_params["strategy"] = MCMCStrategy(verbose=True)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        self.local_rank = local_rank
        self.world_rank = world_rank
        self.world_size = world_size

    def train(self, sfm_dir: Path) -> Path:
        """
        Executes the Gaussian Splatting training.

        Args:
            sfm_dir (Path): The directory containing the COLMAP data (output from ReconstructionPipeline).

        Returns:
            Path: The path to the training results directory.
        """
        logging.info(f"Starting Gaussian Splatting training for data in: {sfm_dir}")

        # Create Config object
        self.config_params["data_dir"] = str(sfm_dir)
        cfg = Config(**self.config_params)
        cfg.adjust_steps(cfg.steps_scaler)

        # Handle conditional imports for BilateralGrid as in simple_trainer.py
        if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
            try:
                if cfg.use_fused_bilagrid:
                    from fused_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
                else:
                    from lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
            except ImportError:
                logging.error("BilateralGrid dependencies not found. Please install them if you intend to use bilateral grid.")
                raise

        # Call the main training function directly
        try:
            simple_trainer_main(
                local_rank=self.local_rank,
                world_rank=self.world_rank,
                world_size=self.world_size,
                cfg=cfg,
            )
        except Exception as e:
            logging.error(f"Error during Gaussian Splatting training: {e}")
            raise

        final_result_path = Path(cfg.result_dir)
        logging.info(f"Gaussian Splatting training finished. Results in: {final_result_path}")
        return final_result_path
