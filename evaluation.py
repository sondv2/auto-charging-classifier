from datetime import datetime


def evaluate_model(model, generator, nBatches=60):
    score = model.evaluate_generator(generator=generator,           # Generator yielding tuples
                                     steps=len(generator) // nBatches)   # number of steps (batches of samples) to yield from generator before stopping

    print("%s: Model evaluated:"
          "\n\t\t\t\t\t\t Loss: %.3f"
          "\n\t\t\t\t\t\t Accuracy: %.3f" %
          (datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
           score[0], score[1]))

def evaluate(modelTrained, validation_generator, train_generator, batch_size):
    # Evaluate on validation data
    print("%s: Model evaluation (valX, valY):" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    evaluate_model(modelTrained, validation_generator, nBatches=batch_size)

    # Evaluate on training data
    print("%s: Model evaluation (trainX, trainY):" % datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    evaluate_model(modelTrained, train_generator, nBatches=batch_size)