import torch
import numpy as np
import matplotlib.pyplot as plt
from homework4 import CNP, bezier, Hw5Env


class CNMP(CNP):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, condition_dim, min_std=0.1):

        super(CNMP, self).__init__(in_shape, hidden_size, num_hidden_layers, min_std)
        self.condition_dim = condition_dim
        del self.query
        
        # Create new
        query_layers = []
        query_layers.append(torch.nn.Linear(hidden_size + self.d_x + condition_dim, hidden_size))
        query_layers.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            query_layers.append(torch.nn.Linear(hidden_size, hidden_size))
            query_layers.append(torch.nn.ReLU())
        query_layers.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*query_layers)

    def forward(self, observation, target, condition, observation_mask=None):
        # Use the encoder and aggregator from CNP
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        
        h_cat = self.concatenate_with_condition(r, target, condition)
        
        query_out = self.decode(h_cat)
        
        # Split into mean and std
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        
        return mean, std
    
    def concatenate_with_condition(self, r, target, condition):
        num_target_points = target.shape[1]
        
        r_repeat = r.unsqueeze(1).repeat(1, num_target_points, 1)
        
        # Repeat condition for each target point
        condition_repeat = condition.unsqueeze(1).repeat(1, num_target_points, 1)
        
        h_cat = torch.cat([r_repeat, target, condition_repeat], dim=-1)
        
        return h_cat
    
    def nll_loss(self, observation, target, target_truth, condition, observation_mask=None, target_mask=None):
       
        mean, std = self.forward(observation, target, condition, observation_mask)
        
        # Calculate NLL loss
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        
        if target_mask is not None:
            # Apply mask for batch processing
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        
        return loss


def collect_data(num_trajectories=100, render_mode="headless"):
    # Collect trajectory data using the robot simulation environment.
    env = Hw5Env(render_mode=render_mode)
    trajectories = []
    
    for i in range(num_trajectories):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        states = np.stack(states)
        
        times = np.linspace(0, 1, states.shape[0]).reshape(-1, 1)
        
        # Each state contains [ey, ez, oy, oz, h]
        trajectory = np.concatenate([times, states], axis=1)
        trajectories.append(trajectory)
        
        print(f"Collected {i+1} trajectories.", end="\r")
    
    return trajectories


def prepare_dataset(trajectories):
    dataset = []
    for traj in trajectories:
        time = traj[:, 0:1]
        ee_pos = traj[:, 1:3]
        obj_pos = traj[:, 3:5]
        height = traj[0, 5]
        
        sm = np.concatenate([ee_pos, obj_pos], axis=1)
        
        dataset.append({
            'time': time,
            'sm': sm,
            'height': height
        })
    
    return dataset


def train_cnmp(dataset, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # CNMP parameters
    d_x = 1
    d_y = 4
    condition_dim = 1
    hidden_size = 128
    num_hidden_layers = 3
    
    # Create model
    model = CNMP(
        in_shape=[d_x, d_y],
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        condition_dim=condition_dim
    ).to(device)
    
    # Training parameters
    num_epochs = 100
    batch_size = 16
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data
        np.random.shuffle(dataset)
        
        # Process in batches
        for batch_start in range(0, len(dataset), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_data = dataset[batch_start:batch_end]
            actual_batch_size = len(batch_data)
            
            # Process each trajectory in the batch
            batch_losses = []
            for data in batch_data:
                time = data['time']
                sm = data['sm']
                height = data['height']
                
                n_context = np.random.randint(1, len(time) + 1)
                context_indices = np.random.choice(len(time), n_context, replace=False)
                
                target_indices = np.arange(len(time))
                
                context_time = torch.tensor(time[context_indices], dtype=torch.float32).to(device)
                context_sm = torch.tensor(sm[context_indices], dtype=torch.float32).to(device)
                
                context = torch.cat([context_time, context_sm], dim=1).unsqueeze(0)
                
                target_time = torch.tensor(time[target_indices], dtype=torch.float32).unsqueeze(0).to(device)
                
                target_sm = torch.tensor(sm[target_indices], dtype=torch.float32).unsqueeze(0).to(device)
                
                condition = torch.tensor([[height]], dtype=torch.float32).to(device)
                
                loss = model.nll_loss(context, target_time, target_sm, condition)
                
                batch_losses.append(loss.item())
                
                loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Calculate average loss for this batch
            avg_batch_loss = np.mean(batch_losses)
            epoch_loss += avg_batch_loss
            num_batches += 1
        
        # Calculate average epoch loss
        epoch_loss /= num_batches
        losses.append(epoch_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNMP Training Loss')
    plt.savefig('cnmp_training_loss.png')
    
    return model


def test_cnmp(model, test_data, n_tests=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    
    ee_errors = []
    obj_errors = []
    
    for _ in range(n_tests):
        # Randomly select a trajectory
        traj_idx = np.random.randint(0, len(test_data))
        trajectory = test_data[traj_idx]
        
        # Extract data
        time = trajectory['time']
        sm = trajectory['sm']
        height = trajectory['height']
        
        total_points = len(time)
        n_context = np.random.randint(1, total_points)
        n_target = np.random.randint(1, total_points)
        
        context_indices = np.random.choice(total_points, n_context, replace=False)
        target_indices = np.random.choice(total_points, n_target, replace=False)
        
        context_time = torch.tensor(time[context_indices], dtype=torch.float32).to(device)
        context_sm = torch.tensor(sm[context_indices], dtype=torch.float32).to(device)
        context = torch.cat([context_time, context_sm], dim=1).unsqueeze(0)
        
        target_time = torch.tensor(time[target_indices], dtype=torch.float32).unsqueeze(0).to(device)
        target_sm = torch.tensor(sm[target_indices], dtype=torch.float32).unsqueeze(0).to(device)
        
        condition = torch.tensor([[height]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            mean, _ = model(context, target_time, condition)
        
        predictions = mean.cpu().numpy()[0]
        ground_truth = target_sm.cpu().numpy()[0]
        
        ee_mse = np.mean((predictions[:, 0:2] - ground_truth[:, 0:2]) ** 2)
        obj_mse = np.mean((predictions[:, 2:4] - ground_truth[:, 2:4]) ** 2)
        
        ee_errors.append(ee_mse)
        obj_errors.append(obj_mse)
    
    ee_error_mean = np.mean(ee_errors)
    ee_error_std = np.std(ee_errors)
    obj_error_mean = np.mean(obj_errors)
    obj_error_std = np.std(obj_errors)
    
    print(f"End-effector MSE: {ee_error_mean:.6f} ± {ee_error_std:.6f}")
    print(f"Object MSE: {obj_error_mean:.6f} ± {obj_error_std:.6f}")
    
    labels = ['End-effector', 'Object']
    means = [ee_error_mean, obj_error_mean]
    errors = [ee_error_std, obj_error_std]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, means, yerr=errors, capsize=10, alpha=0.7, color=['blue', 'red'])
    plt.ylabel('Mean Squared Error')
    plt.title('CNMP Test Results: Prediction Errors')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig('cnmp_test_errors.png')
    plt.show()
    
    return ee_error_mean, ee_error_std, obj_error_mean, obj_error_std


def visualize_predictions(model, test_data, num_samples=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        traj_idx = np.random.randint(0, len(test_data))
        trajectory = test_data[traj_idx]
        
        time = trajectory['time']
        sm = trajectory['sm']
        height = trajectory['height']
        
        total_points = len(time)
        n_context = max(3, int(0.3 * total_points))
        context_indices = np.random.choice(total_points, n_context, replace=False)
        
        target_indices = np.arange(total_points)
        
        context_time = torch.tensor(time[context_indices], dtype=torch.float32).to(device)
        context_sm = torch.tensor(sm[context_indices], dtype=torch.float32).to(device)
        context = torch.cat([context_time, context_sm], dim=1).unsqueeze(0)
        
        target_time = torch.tensor(time[target_indices], dtype=torch.float32).unsqueeze(0).to(device)
        
        condition = torch.tensor([[height]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            mean, std = model(context, target_time, condition)
        
        mean_np = mean.cpu().numpy()[0]
        std_np = std.cpu().numpy()[0]
        
        plt.subplot(num_samples, 2, 2*i+1)
        
        plt.scatter(time[context_indices], sm[context_indices, 0], c='blue', marker='o', label='Context ey')
        plt.scatter(time[context_indices], sm[context_indices, 1], c='green', marker='o', label='Context ez')
        
        plt.plot(time, sm[:, 0], 'b--', alpha=0.5, label='Ground truth ey')
        plt.plot(time, sm[:, 1], 'g--', alpha=0.5, label='Ground truth ez')
        
        plt.plot(time, mean_np[:, 0], 'b-', label='Predicted ey')
        plt.plot(time, mean_np[:, 1], 'g-', label='Predicted ez')
        
        plt.fill_between(
            time.flatten(), 
            mean_np[:, 0] - 2 * std_np[:, 0], 
            mean_np[:, 0] + 2 * std_np[:, 0], 
            color='blue', alpha=0.2
        )
        plt.fill_between(
            time.flatten(), 
            mean_np[:, 1] - 2 * std_np[:, 1], 
            mean_np[:, 1] + 2 * std_np[:, 1], 
            color='green', alpha=0.2
        )
        
        plt.title(f'End-effector Trajectory (Object Height: {height:.3f})')
        plt.xlabel('Time')
        plt.ylabel('Position')
        if i == 0:
            plt.legend()
        
        plt.subplot(num_samples, 2, 2*i+2)
        
        plt.scatter(time[context_indices], sm[context_indices, 2], c='red', marker='o', label='Context oy')
        plt.scatter(time[context_indices], sm[context_indices, 3], c='purple', marker='o', label='Context oz')
        
        plt.plot(time, sm[:, 2], 'r--', alpha=0.5, label='Ground truth oy')
        plt.plot(time, sm[:, 3], 'm--', alpha=0.5, label='Ground truth oz')
        
        plt.plot(time, mean_np[:, 2], 'r-', label='Predicted oy')
        plt.plot(time, mean_np[:, 3], 'm-', label='Predicted oz')
        
        plt.fill_between(
            time.flatten(), 
            mean_np[:, 2] - 2 * std_np[:, 2], 
            mean_np[:, 2] + 2 * std_np[:, 2], 
            color='red', alpha=0.2
        )
        plt.fill_between(
            time.flatten(), 
            mean_np[:, 3] - 2 * std_np[:, 3], 
            mean_np[:, 3] + 2 * std_np[:, 3], 
            color='purple', alpha=0.2
        )
        
        plt.title(f'Object Trajectory (Object Height: {height:.3f})')
        plt.xlabel('Time')
        plt.ylabel('Position')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('cnmp_visualizations.png')
    plt.show()


def train():
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        print("Loading existing trajectories...")
        trajectories = np.load('trajectories.npy', allow_pickle=True)
    except FileNotFoundError:
        print("Collecting new trajectories...")
        trajectories = collect_data(num_trajectories=100)
        np.save('trajectories.npy', trajectories)
    
    dataset = prepare_dataset(trajectories)
    print(f"Dataset contains {len(dataset)} trajectories")
    
    np.random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    print(f"Training set: {len(train_data)} trajectories")
    print(f"Test set: {len(test_data)} trajectories")
    
    model = train_cnmp(train_data)
    
    torch.save(model.state_dict(), 'cnmp_model.pt')
    print("Model trained and saved to 'cnmp_model.pt'")
    
    return model, test_data


def test():
    torch.manual_seed(42)
    np.random.seed(42)
    
    d_x = 1
    d_y = 4
    condition_dim = 1
    hidden_size = 128
    num_hidden_layers = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = CNMP(
        in_shape=[d_x, d_y],
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        condition_dim=condition_dim
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('cnmp_model.pt', map_location=device))
        print("Loaded trained model from 'cnmp_model.pt'")
    except FileNotFoundError:
        print("No trained model found at 'cnmp_model.pt'. Please train the model first.")
        return
    
    try:
        print("Loading existing trajectories...")
        trajectories = np.load('trajectories.npy', allow_pickle=True)
    except FileNotFoundError:
        print("No trajectories found. Please train the model first.")
        return
    
    dataset = prepare_dataset(trajectories)
    print(f"Dataset contains {len(dataset)} trajectories")
    
    np.random.seed(42)
    np.random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    test_data = dataset[train_size:]
    print(f"Test set: {len(test_data)} trajectories")
    
    test_cnmp(model, test_data, n_tests=100)
    
    visualize_predictions(model, test_data, num_samples=3)


if __name__ == "__main__":
    # Uncomment for train or test
    train()
    # test() 