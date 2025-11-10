"""波形付きトレイの G-code を決定的に生成するスクリプト."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import gcoordinator as gc

# GUI エクスポートで扱う Path / PathList をまとめたエイリアス
TrayPath = Union[gc.Path, gc.PathList]

def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """ロジスティックシグモイド σ(x)=1/(1+e^{-x}) を返す."""
    return 1.0 / (1.0 + np.exp(-x))

@dataclass(frozen=True)
class WaveTrayParams:
    """波形トレイのプロポーションやインフィル条件を集約する設定."""

    layer_count: int = 120  # 積層数
    layer_height_mm: float = 1.0  # レイヤー厚 [mm]
    base_radius_mm: float = 20  # ベース半径 [mm]
    radial_growth_mm: float = 5  # 高さに応じた半径変化幅
    wave_amplitude_mm: float = 2.0  # 波形の追加振幅
    wave_frequency: float = 100.5  # 波形周波数
    wave_sample_count: int = 403  # 波形用サンプル数
    hole_sample_count: int = 200  # 中央ホールのサンプル数
    hole_radius_mm: float = 15.3  # 中央ホール半径 [mm]
    hole_layer_limit: int = 2  # 何層までホールを開けるか
    hole_offset_mm: float = 0.4  # ホールの内側オフセット
    infill_distance_mm: float = 0.5
    infill_angle_offset_rad: float = np.pi / 4
    infill_angle_step_rad: float = np.pi / 2
    concentric_pitch_mm: float = 1.0  # 底面インフィルのリング間隔
    concentric_clearance_mm: float = 0.2  # 内外輪郭からの逃げ寸法


def build_concentric_fill(
    params: WaveTrayParams, inner_radius: float, outer_radius: float, z_height: float
) -> gc.PathList | None:
    """内外半径の間を同心円パスで満たす."""

    if outer_radius - inner_radius <= params.concentric_pitch_mm:
        return None

    angles = np.linspace(0, 2 * np.pi, params.hole_sample_count)
    z_coords = np.full_like(angles, z_height, dtype=float)
    radii = np.arange(
        inner_radius, outer_radius, params.concentric_pitch_mm, dtype=float
    )

    paths: list[gc.Path] = []
    for radius in radii:
        path = gc.Path(radius * np.cos(angles), radius * np.sin(angles), z_coords)
        paths.append(path)

    return gc.PathList(paths) if paths else None


def build_wave_tray(params: WaveTrayParams) -> list[TrayPath]:
    """層ごとの経路を生成して返す。

    1. 外周の波形リムを生成
    2. 指定層まで中央ホールと内側オフセットを作成
    3. 波形とホールを輪郭とするインフィルを敷設
    """
    tray_paths: list[TrayPath] = []
    # 外周波形とホール用に角度配列を事前生成
    wave_angles = np.linspace(0, 2 * np.pi, params.wave_sample_count)
    hole_angles = np.linspace(0, 2 * np.pi, params.hole_sample_count)
    # 全体高さで正規化するための値
    total_height_mm = params.layer_count * params.layer_height_mm

    for layer_index in range(params.layer_count):
        # レイヤーごとの高さレンジと正規化高さを算出
        layer_bottom = layer_index * params.layer_height_mm
        layer_top = (layer_index + 1) * params.layer_height_mm
        layer_z = np.linspace(layer_bottom, layer_top, params.wave_sample_count)
        normalized_height = layer_z / total_height_mm

        # 高さに応じてベース径を膨らませ、縦方向のうねりを作る
        base_radius = params.base_radius_mm + params.radial_growth_mm * sigmoid(6*(normalized_height - 0.5)
        )
        # ベース径に波形を重ねて外周のリム形状を作る
        wave_radius = base_radius + params.wave_amplitude_mm * np.sin(
            wave_angles * params.wave_frequency + np.pi * layer_index
        )
        wave_wall = gc.Path(
            wave_radius * np.cos(wave_angles),
            wave_radius * np.sin(wave_angles),
            layer_z,
        )
        tray_paths.append(wave_wall)

        if layer_index < params.hole_layer_limit:
            # ホールは各層で一定高さの円として追加
            hole_z = np.full_like(
                hole_angles, params.layer_height_mm * (layer_index + 1), dtype=float
            )
            hole_path = gc.Path(
                params.hole_radius_mm * np.cos(hole_angles),
                params.hole_radius_mm * np.sin(hole_angles),
                hole_z,
            )
            tray_paths.append(hole_path)

            # ホールの壁厚を確保するため内側オフセットを生成
            inner_hole = gc.Transform.offset(hole_path, -params.hole_offset_mm)
            tray_paths.append(inner_hole)

            # 底面は同心円インフィルに置き換える
            inner_radius = params.hole_radius_mm + params.concentric_clearance_mm
            outer_radius = float(wave_radius.min()) - params.concentric_clearance_mm
            concentric = build_concentric_fill(
                params=params,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                z_height=params.layer_height_mm * (layer_index + 1),
            )
            if concentric is not None:
                concentric.z_hop = False
                concentric.retraction = False
                tray_paths.append(concentric)

    return tray_paths


if __name__ == "__main__":
    default_params = WaveTrayParams()
    # デフォルト設定で経路を生成して GUI へ渡す
    gc.gui_export(build_wave_tray(default_params))
