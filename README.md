# Project Setup Guide

## Setting Up the GitHub Environment

Follow these steps to set up the project environment after cloning the repository:

### 1. Clone the Repository

```sh
git clone https://github.com/Digital-Twin-GP/RL-CARLA-TRAINING.git
cd RL-CARLA-TRAINING
```

### 2. Create and Activate a Python Virtual Environment (Python 3.7.2)

#### Windows:
```sh
py -3.7 -m venv carla_env
carla_env\Scripts\activate
```

#### Linux:
```sh
python3.7 -m venv carla_env
source carla_env/bin/activate
```

### 3. Upgrade pip

```sh
python -m pip install --upgrade pip
```

### 4. Install Required Dependencies

```sh
pip install -r requirements.txt
```

### 5. Run Carla Simulator

Ensure the Carla simulator (`CarlaUE4`) is running before executing any scripts.

#### Windows:
```sh
CarlaUE4.exe
```

#### Linux:
```sh
./CarlaUE4.sh
```

### 6. Running Project Modes

Once `CarlaUE4` is running, you can execute any Python script in the project directory. The project supports three main modes:

#### 1. Normal Simulation Mode

Runs the standard Carla simulation with the default agent.

```sh
python main.py
```

#### 2. RL Training Mode

Trains a reinforcement learning agent for fuel efficiency.

- **SAC (default):**
  ```sh
  python main.py --rl
  ```
- **DQN:**
  ```sh
  python main.py --rl --dqn
  ```

#### 3. Test Trained Model

Tests a trained RL model against the BasicAgent.

```sh
python main.py --test --model-path <path_to_model>
```
Replace `<path_to_model>` with your trained model file, e.g.:
```sh
python main.py --test --model-path models/FuelOptimizer__1749141785.model
```
#### 4. RL Inference Interface Mode

Use this mode to run the RL inference interface as a standalone script. This allows you to get throttle predictions from a trained RL model (SAC by default, or DQN) using speed and acceleration as input. This is useful for distributed setups where CARLA and RL inference run on separate machines.

**Example usage:**

```sh
python rl_inference_interface.py --model_path <path_to_model> --agent_type <type> --speed <speed_value> --acceleration <acceleration_value>
```

Replace `<path_to_model>`, `<speed_value>`, and `<acceleration_value>` with your actual values.

---

## Viewing Model Logs with TensorBoard

To visualize logs using TensorBoard, run:

```sh
tensorboard --logdir=logs/
```

Then open your browser at:

```
http://localhost:6006
```

This will display training logs and model performance metrics.

### Notes:
- Ensure the virtual environment is activated before running scripts.
- Make sure `CarlaUE4` is running before executing any Python files.