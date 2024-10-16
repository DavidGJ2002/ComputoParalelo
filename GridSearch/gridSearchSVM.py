import itertools
import multiprocess
import time 
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

# Método para hacer nivelación de cargas
def nivelacion_cargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

# Parámetros para SVM
param_grid_svm = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Generar combinaciones para KNN
keys_svm, values_svm = zip(*param_grid_svm.items())
combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

# Función a paralelizar
def evaluate_set(hyperparameter_set, lock):
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import fetch_openml
    """
    Evaluate a set of hyperparameters for SVM
    """
    # Cargar MNIST
    mnist = fetch_openml('mnist_784', parser='auto')
    X = mnist.data[:10000]
    y = mnist.target[:10000]

    # Convertir las etiquetas a enteros
    y = y.astype(int)

    # Convertir los datos a arrays de numpy
    X = np.array(X)
    y = np.array(y)

    # Particionar el conjunto de datos en 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)

    timeI = time.perf_counter()
    
    for s in hyperparameter_set:
        clf = SVC()
        clf.set_params(C=s['C'], kernel=s['kernel'], gamma=s['gamma'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Exclusión mutua
        lock.acquire()
        print('Accuracy en el proceso:', accuracy_score(y_test, y_pred))
        lock.release()

    timeT = time.perf_counter() - timeI
    print(f'Tiempo transcurrido: {timeT} segundos')

if __name__ == '__main__':
    threads = []
    N_THREADS = 4
    splits = nivelacion_cargas(combinations_svm, N_THREADS)
    lock = multiprocess.Lock()
    
    for i in range(N_THREADS):
        threads.append(multiprocess.Process(target=evaluate_set, args=(splits[i], lock)))

    start_time = time.perf_counter()
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
                
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
