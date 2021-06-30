from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def post_process(model, validation_generator, his):
    scores = model.evaluate_generator(generator=validation_generator, steps=64)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
    Y_pred = model.predict_generator(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(validation_generator.classes, y_pred))

    ground = validation_generator.classes
    print(classification_report(ground, y_pred))

    get_acc = his.history['accuracy']
    value_acc = his.history['val_accuracy']
    get_loss = his.history['loss']
    validation_loss = his.history['val_loss']

    epochs = range(len(get_acc))
    plt.plot(epochs, get_acc, 'r', label='Accuracy of Training data')
    plt.plot(epochs, value_acc, 'b', label='Accuracy of Validation data')
    plt.title('Training vs validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs, get_loss, 'r', label='Loss of Training data')
    plt.plot(epochs, validation_loss, 'b', label='Loss of Validation data')
    plt.title('Training vs validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
