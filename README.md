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

### 6. Running Python Scripts

Once `CarlaUE4` is running, you can execute any Python script in the project directory.

#### Example:
```sh
python carla_fuel_consumption_initial.py
```

### 7. Testing the Trained Model

To test the output model, run the following script:
```sh
python carla_trained_model_test.py
```
Before running, update the `MODEL_PATH` variable in `carla_trained_model_test.py`:
```python
MODEL_PATH = 'models/Xception___-26.65max_-202.85avg_-331.95min__1741692714.model'  # Update with your model name
```

## Viewing Model Logs with TensorBoard

To visualize logs using TensorBoard, run the following command:

```sh
tensorboard --logdir=logs/
```

Then, open a web browser and visit:

```
http://localhost:6006
```

This will display training logs and model performance metrics.

### Notes:
- Ensure the virtual environment is activated before running scripts.
- Make sure `CarlaUE4` is running before executing any Python files.