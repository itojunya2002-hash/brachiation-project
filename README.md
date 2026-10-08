# MuJoCo 2-Link Brachiation Control with Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-orange.svg)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-v3.0%2B-green.svg)](https://mujoco.org/)

物理エンジン **MuJoCo** と深層強化学習（**PPO / Stable-Baselines3**）を用いた、2リンクブランキエーションロボットの目標位置スイング制御シミュレーションフレームワークです。

## 🌟 Advanced Features (Sim-to-Sim Transfer)
- **剛体から柔軟構造（弾性枝）への転移学習 (`envs/brachiation_transfer_env.py`)**: 
  現実世界の構造物を模した「しなり（弾性変形）」をモデル化。剛体モデルで獲得した基礎方策（Base Policy）を初期値とし、柔軟環境下でファインチューニングを行うことで、不安定なダイナミクスに対するロバストな制御方策を獲得します。
- **統一された観測空間設計**: 
  剛体・弾性環境間で共通の低次元観測空間を設計し、異なる物理モデル間でのスムーズなモデル読み込み・転移を実現しています。

> **⚠️ Note on Intellectual Property:**
> 本リポジトリはポートフォリオ用のモジュール設計・環境構築のサンプルです。現在研究中の特定の報酬関数チューニングおよびハイパーパラメータの詳細な数値は、知財保護および学会発表準備の観点から非公開としており、一般化したプレースホルダー構造に置き換えています。

---

## 📂 Project Architecture

brachiation-project/
├── envs/                     # 環境バリエーション・転移学習用環境
│   ├── brachiation_env.py    # 標準物理環境
│   └── brachiation_transfer_env.py # 弾性枝（柔軟構造）転移用環境
├── models/                   # 学習済みモデル（重みデータ）保存用
├── ppo_log/                  # 学習ログ（TensorBoard等）
├── pendulum_dynamics.py      # 物理演算・力学モデルのインターフェース
├── pendulum_leap_train.py    # PPOを用いた強化学習の学習スクリプト
├── enjoy.py                  # 学習済みモデルの動作確認・可視化スクリプト
└── analyze_physics.py        # 物理量（角速度・トルク等）の解析・プロット

---

## 🚀 主な特徴 (Key Features)
1. **マルチフェーズ制御環境 (Multi-Phase Control)**
   - スイングアップ、空中飛行フェーズ、ターゲットへの非弾性衝突・制振処理を統合。
2. **クリーンなモジュール設計**
   - 物理ダイナミクス計算とGymnasium環境クラスを完全に分離し、拡張性の高いコードベースを実現。
3. **シームレスなSim-to-Simパイプライン**
   - 標準環境での基礎訓練から、発展的な環境（`envs/`）への転移学習が容易な設計。

---

## 📦 必要環境 (Requirements)
* Python 3.10以上
* ライブラリのインストール:
```bash
pip install -r requirements.txt