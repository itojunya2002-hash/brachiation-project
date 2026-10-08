# MuJoCo 2-Link Brachiation Control with Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-orange.svg)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-v3.0%2B-green.svg)](https://mujoco.org/)

物理エンジン **MuJoCo** と深層強化学習（**PPO / Stable-Baselines3**）を用いた、2リンクブランキエーションロボットの目標位置スイング制御シミュレーションフレームワークです。

## Advanced Features (Sim-to-Sim Transfer)
- **剛体から柔軟構造（弾性枝）への転移学習**: 
  現実世界の構造物を模した「しなり（弾性変形）」をスライドジョイントでモデル化。剛体モデルで獲得した基礎方策（Base Policy）を初期値とし、柔軟環境下でファインチューニングを行うことで、不安定なダイナミクスに対するロバストな制御方策を獲得。
- **統一された観測空間設計**: 
  剛体・弾性環境間で共通の10次元観測空間（$cos, sin, v, t, y_b$）を設計し、異なる物理モデル間でのスムーズなモデル読み込み・転移を実現。


> ** Note on Intellectual Property:**
> 本リポジトリはポートフォリオ用のモジュール設計・環境構築のサンプルです。現在研究中の特定の報酬関数チューニングおよびハイパーパラメータの詳細な数値は、知財保護および学会発表準備の観点から非公開としており、一般化したプレースホルダー構造に置き換えています。

---

## 📂 Project Architecture