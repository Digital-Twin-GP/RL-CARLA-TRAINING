import numpy as np
import argparse
from tensorflow.keras.models import load_model

class RLInference:
    def __init__(self, model_path, agent_type='sac'):
        self.agent_type = agent_type.lower()
        self.model = load_model(model_path, compile=False)

    def predict_throttle(self, speed, acceleration):
        state = np.array([speed, acceleration], dtype=np.float32).reshape(1, -1)
        if self.agent_type == 'dqn':
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])
            throttle = [0.2, 0.5, 1.0][action]
        elif self.agent_type == 'sac':
            mu, _ = self.model(state)
            throttle = float(mu.numpy()[0][0])
        else:
            raise ValueError("Unknown agent_type. Use 'dqn' or 'sac'.")
        return throttle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Inference Interface")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--agent_type', type=str, choices=['dqn', 'sac'], default='sac', help='Type of RL agent')
    parser.add_argument('--speed', type=float, required=True, help='Current speed (km/h)')
    parser.add_argument('--acceleration', type=float, required=True, help='Current acceleration (m/s^2)')
    args = parser.parse_args()

    rl_infer = RLInference(args.model_path, args.agent_type)
    throttle = rl_infer.predict_throttle(args.speed, args.acceleration)
    print(f"Predicted throttle: {throttle:.3f}")