# MuJoCo 2-Link Brachiation Control with Reinforcement Learning (Rigid & Hybrid Framework)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-orange.svg)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-v3.0%2B-green.svg)](https://mujoco.org/)

物理エンジンおよびカスタム力学モデルと深層強化学習（**PPO / Stable-Baselines3**）を用いた、2リンクブランキエーションロボットの**剛体枝環境での基礎跳躍動作〜弾性枝（柔軟構造）環境への展開・空中飛行・ターゲット把持（制振）**に至る一連の制御シミュレーションフレームワークです。

## 🌟 Advanced Features & Key Concepts
- **剛体枝モデルから弾性枝（柔軟構造）への発展**: 
  まず固定ベースや標準的な剛体枝環境（`custom_numpy_env.py` / `pendulum_dynamics.py`）で跳躍・スイングアップの基礎方策を構築・検証。その知見をベースに、現実世界の構造物が持つ「たわみ・ばね弾性（Spring-Damper）」を組み込んだハイブリッド環境（`custom_numpy_env_elastic.py`）へ拡張し、不安定なダイナミクスに対するロバストな制御を獲得します。
- **マルチフェーズ・ハイブリッド制御**: 
  - **Phase 1 (Swing):** 枝上でのスイングアップと最適なリリースタイミングの探索
  - **Phase 2 (Flight):** 自由浮遊系（重心の放物運動と重心周りの回転運動）に基づく空中飛行
  - **Phase 3 (Catch & Damp):** ターゲット把持時の非弾性衝突・衝撃緩和と速やかな制振処理
- **統一された観測空間・モジュール設計**: 
  物理演算ロジック（Dynamics）とGymnasium環境クラスを完全に分離し、ハードウェア検証用スクリプト等も含めて拡張性の高いクリーンなアーキテクチャを採用しています。

> **⚠️ Note on Intellectual Property:**
> 本リポジトリはポートフォリオ用のモジュール設計・環境構築のサンプルです。現在研究中の特定の報酬関数チューニング、バネ・ダンパー係数の詳細な実機同定値、およびハイパーパラメータの詳細な数値は、知財保護および学会発表準備の観点から非公開としており、一般化したプレースホルダー構造に置き換えています。

---

## 📂 Project Architecture

brachiation-project/
├── envs/                           # 各種環境定義フォルダ
│   ├── brachiation_env.py          # 標準物理環境（剛体ベース）
│   ├── brachiation_transfer_env.py # （剛体ベース→弾性ベース）転移学習・拡張用環境
│   └── hardware_consideration/     # 単リンク等のハードウェア考慮・検証用スクリプト
├── models/                         # 学習済みモデル（重みデータ）保存用
├── ppo_log/                        # 剛体モデルの学習ログ
├── logs_hybrid_branch/             # 弾性枝モデルの学習ログ
├── pendulum_dynamics.py            # 剛体枝での跳躍ベースの物理演算・力学モデル
├── custom_numpy_env.py             # 弾性枝での跳躍ベースのGymnasium環境本体
├── pendulum_leap_train.py          # 剛体枝での跳躍モデルの学習用メインスクリプト
├── pendulum_elastic_train.py       # 弾性枝での跳躍モデルの学習用メインスクリプト
├── enjoy.py                        # 剛体枝での跳躍モデルの動作確認・可視化
├── enjoy_elastic.py                # 弾性枝での跳躍モデルの動作確認・可視化
├── analyze_physics.py              # 剛体枝での跳躍モデルの物理量解析
└── analyze_physics_elastic.py      # 弾性枝での跳躍モデルの物理量解析

---

## 📦 必要環境 (Requirements)
- Python 3.10以上
- ライブラリのインストール:
```bash
pip install -r requirements.txt