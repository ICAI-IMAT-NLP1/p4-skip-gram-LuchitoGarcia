import torch
import torch.optim as optim
from typing import List, Dict

try:
    from src.skipgram import SkipGramNeg, NegativeSamplingLoss  
    from src.data_processing import get_batches, cosine_similarity
except ImportError:
    from skipgram import SkipGramNeg, NegativeSamplingLoss
    from data_processing import get_batches, cosine_similarity

def train_skipgram(model: SkipGramNeg,
                   words: List[int], 
                   int_to_vocab: Dict[int, str], 
                   batch_size=512, 
                   epochs=5, 
                   learning_rate=0.003, 
                   window_size=5, 
                   print_every=1500,
                   device='cpu'):
    """Trains the SkipGram model using negative sampling.

    Args:
        model: The SkipGram model to be trained.
        words: A list of words (integers) to train on.
        int_to_vocab: A dictionary mapping integers back to vocabulary words.
        batch_size: The size of each batch of input and target words.
        epochs: The number of epochs to train for.
        learning_rate: The learning rate for the optimizer.
        window_size: The size of the context window for generating training pairs.
        print_every: The frequency of printing the training loss and validation examples.
        device: The device (CPU or GPU) where the tensors will be allocated.
    """
    # Mover modelo al dispositivo correspondiente
    model.to(device)

    # Definir la función de pérdida y el optimizador
    criterion = NegativeSamplingLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0

    # Bucle de entrenamiento    
    for epoch in range(epochs):
        # Generar batches con la función get_batches
        for input_words, target_words in get_batches(words, batch_size, window_size):
            steps += 1
           
            inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
            inputs, targets = inputs.to(device), targets.to(device)

            # Obtener vectores de entrada, salida y ruido
            input_vectors = model.forward_input(inputs)
            output_vectors = model.forward_output(targets)
            noise_vectors = model.forward_noise(inputs.size(0), window_size)

            # Calcular la pérdida usando la función de pérdida personalizada
            loss = criterion(input_vectors, output_vectors, noise_vectors)

            # Paso hacia atrás y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Imprimir métricas periódicamente
            if steps % print_every == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Step: {steps}, Loss: {loss.item():.4f}")

                # Calcular similitud coseno para palabras de validación
                valid_examples, valid_similarities = cosine_similarity(model.out_embed, device=device)
                _, closest_idxs = valid_similarities.topk(6)  # Tomamos las 6 más cercanas

                valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')

                for i, valid_idx in enumerate(valid_examples):
                    valid_word = int_to_vocab[valid_idx.item()]
                    closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[i][1:]]  # Excluir la palabra en sí
                    print(f"{valid_word} | {', '.join(closest_words)}")

                print("...\n")

