import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader
from src.dataset import FEMNISTDataset
from src.genetic import GeneticAlgorithm
from src.train import train_and_evaluate_model
from src.models import CNN
import numpy as np
import copy

if __name__ == "__main__":
    print("Loading FEMNIST dataset...")

    ds = load_dataset("flwrlabs/femnist")
    transform = T.Compose([
        T.Resize((28,28)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

    temp_split = ds['train'].train_test_split(test_size=0.1, seed=42)
    test_split = temp_split['test']
    train_valid = temp_split['train'].train_test_split(test_size=0.1, seed=42)

    ga_train_size = 20000
    ga_val_size = 5000

    ga_train_data = train_valid['train'].select(range(min(ga_train_size, len(train_valid['train']))))
    ga_val_data = train_valid['test'].select(range(min(ga_val_size, len(train_valid['test']))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ga = GeneticAlgorithm(population_size=6, mutation_rate=0.3, crossover_rate=0.7)
    population = ga.create_population()

    generations = 5
    best_fitness_history = []
    best_individual = None
    best_fitness = 0.0

    print(f"Starting GA optimization with {len(population)} individuals for {generations} generations...")
    for generation in range(generations):
        print(f"\n---Generation {generation + 1}/{generations}---")
        fitness_scores = []

        for i, individual in enumerate(population):
            print(f"Evaluating individual {i+1}/{len(population)}...")
            train_ds = FEMNISTDataset(ga_train_data, transform)
            val_ds = FEMNISTDataset(ga_val_data, transform)
            train_loader = DataLoader(train_ds, batch_size=individual.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=individual.batch_size, shuffle=False, num_workers=2)
            fitness = train_and_evaluate_model(individual, train_loader, val_loader, device)
            fitness_scores.append(fitness)
            print(f"->Fitness: {fitness:.4f}")
            print(f"->Params: learning_rate={individual.learning_rate:.4f}, weight_decay={individual.weight_decay:.6f}, "
                  f"dropout_rate={individual.dropout_rate:.3f}, width_mult={individual.width_mult:.3f}")
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = copy.deepcopy(individual)

        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        best_fitness_history.append(max_fitness)

        print(f"Generation {generation + 1} - Avg: {avg_fitness:.4f}, Best: {max_fitness:.4f}")
        if generation < generations - 1:
            selected_parents = ga.select_parents(population, fitness_scores)
            new_population = []
            best_idx = np.argmax(fitness_scores)
            new_population.append(copy.deepcopy(population[best_idx]))

            while len(new_population) < ga.population_size:
                parent1, parent2 = np.random.choice(selected_parents, 2, replace=False)
                if np.random.rand() < ga.crossover_rate:
                    child1, child2 = ga.crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                child1 = ga.mutate(child1)
                child2 = ga.mutate(child2)
                new_population.extend([child1, child2])
            population = new_population[:ga.population_size]

    print(f"\n---GA Optimization Done---")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Best hyperparams:")
    print(f"->Learning rate: {best_individual.learning_rate:.6f}")
    print(f"->Weight decay: {best_individual.weight_decay:.6f}")
    print(f"->Dropout rate: {best_individual.dropout_rate:.4f}")
    print(f"->Width multiplier: {best_individual.width_mult:.4f}")
    print(f"->Batch size: {best_individual.batch_size}")
    print(f"->Momentum: {best_individual.momentum:.4f}")

    full_train_ds = FEMNISTDataset(train_valid['train'], transform)
    full_val_ds = FEMNISTDataset(train_valid['test'], transform)
    full_test_ds = FEMNISTDataset(test_split, transform)

    train_loader = DataLoader(full_train_ds, batch_size=best_individual.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(full_val_ds, batch_size=best_individual.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(full_test_ds, batch_size=best_individual.batch_size, shuffle=False, num_workers=2)

    final_model = CNN(num_classes=62, hyperparams=best_individual).to(device)
    optimizer = torch.optim.SGD(
        final_model.parameters(),
        lr=best_individual.learning_rate,
        momentum=best_individual.momentum,
        weight_decay=best_individual.weight_decay
    )

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print(f"Final model parameters: {sum(p.numel() for p in final_model.parameters())}")
    print(f"\nTraining Model with optimised hyperparams...")
    
    epochs = 10
    best_val_acc = 0.0
    for epoch in range(epochs):
        final_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        pbar = range(len(train_loader))

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = final_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

        final_model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = final_model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)

        val_acc = val_correct / val_total
        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(final_model.state_dict(), "ga_optimized_model.pth")
        scheduler.step()

    final_model.load_state_dict(torch.load("ga_optimized_model.pth", map_location=device))
    final_model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = final_model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
    final_test_acc = test_correct / test_total
    
    print(f"\nResults:")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print(f"Model saved as 'ga_optimized_model.pth'")
