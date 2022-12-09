import numpy as np
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt
import os.path

# TODO: Fit ()
def resize(imagen):
    #Porcentaje en el que se redimensiona la imagen
    scale_percent = 50
    #calcular el 50 por ciento de las dimensiones originales
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # cambiar el tamaño de la image
    output = cv.resize(src, dsize)
    cv.imwrite('jit1.jpg',output)

numsClusters = 4
clusters = []
numCarac = 3
centroids = np.zeros((numsClusters, numCarac))

np.random.seed()

def distanciaEuclidiana(p1, p2):
    n = p1.shape[0]#numero de filas 
    sumCuadrados = 0
    for i in range(n):
        sumCuadrados += (p1[i] - p2[i]) ** 2
    return math.sqrt(sumCuadrados)

def predecirCamino(s):
    #predecir la distancia mas 
    # cercana entre un cluster y un sample
    # otorga un sample a un cluster y retorna el cluster que esta mas cerca del sample
    global centroids
    distanciaMasCercana = float('inf')
    ClusterMasCercano = 0
    for numCluster in range(numsClusters):
        #centroid es tomar uno de los centroides disponibles()
        centroid = centroids[numCluster] # se toma un centoride a la vez
        tmpDistancia = distanciaEuclidiana(centroid, s)
        #si la distancia tmp es menor a la distancia mas cercana
        if tmpDistancia < distanciaMasCercana:
            distanciaMasCercana = tmpDistancia
            ClusterMasCercano = numCluster
    return ClusterMasCercano

def puntosACluster(dataset:np.ndarray):
    #otorgar un cluster a los samples
    global clusters
    clusters = []
    for numCluster in range(numsClusters):
        clusters.insert(numCluster, [])#se guardan los cluster en un arreglo
    for s in dataset:
        #va a determinar el mejor cluster(cercano)
        ClusterMasCercano = predecirCamino(s)
        clusters[ClusterMasCercano].append(s)

def recalculoDeCentroide():
    global clusters
    #se recalculan los centroides a apartir de un promedio de cada feature por cada cluster
    for numCluster in range(numsClusters):
        #clusters append de samples
        cluster = np.array(clusters[numCluster])
        tmpCentroide = []
        if len(cluster) <= 0: #el cluster estaa vacio =0
            continue
        for nFeature in range(numCarac):
            feature_array = cluster[:, nFeature]
            #promedio de feature_array
            tmpCentroide.append(np.average(feature_array))
        #se hacen las comparaciones hasta que los puntos sean origiales
        #np.floor truncar valores
        centroids[numCluster] = np.floor(tmpCentroide)

def agregarCentroideAleatorio(dataset:np.ndarray):
    #Ootorga un centroide escogiendo uno de los n-cluster con samples random y tomandolo como nuevo centroide 
    nS, _ = dataset.shape
    global centroids
    centroids = np.zeros((numsClusters, numCarac))#inicializar en ceros
    for nCluster in range(numsClusters):
        #generar el centroide en una de las cordenadas
        rnd = np.random.randint(0, nS)
        #se van a asignar diferentes centroides a partir de un numero random
        centroids[nCluster] = dataset[rnd]

def Ajustar(dataset:np.ndarray):
    #Entrenar al modelo de k-means a partir de un dataset
    nSamples, nFeaturesL = dataset.shape #detectar numero de filas y columnas del dataset
    global nFeatures
    nFeatures = nFeaturesL # atributo
    i = 0
    agregarCentroideAleatorio(dataset)
    global centroids
    tmpCentroides = np.array([])
    #comparar los centroides y que sean pocas iteraciones
    while (not np.array_equal(tmpCentroides, centroids)) and i < 100:           
        tmpCentroides = centroids.copy()
        puntosACluster(dataset)
        recalculoDeCentroide()
        i += 1

def procesamientoKmeans(imagen):
    if not os.path.exists("kmeans.jpg"):
        img = cv.imread("jit.JPG")
        w,h,c = img.shape
        imgPreproceso = np.reshape(img,(w*h,c))#transformar una lista
        imgProcesada = np.zeros((w,h,c), dtype=np.uint8)
        Ajustar(imgPreproceso)
        for x in range(w):
            for y in range(h):
                cluster_predicted = predecirCamino(img[x][y])
                imgProcesada[x][y] = np.floor(centroids[cluster_predicted])
        print(f"Centroides: {centroids}")
        
        cv.imwrite("kmeans.jpg",imgProcesada)
    return cv.imread("kmeans.jpg")

def separar_colores(img: np.ndarray, color: np.ndarray):
    if not os.path.exists("colores_separados.jpg"):
        w,h,c = img.shape
        res = np.zeros((w, h, c), dtype=np.uint8)
        for x in range(w):
            for y in range(h):
                b,g,r = img[x][y]
                if r == color[0] and g == color[1] and b == color[2]:
                    res[x][y] = 255
        cv.imwrite("colores_separados.jpg",res)
    return cv.imread("colores_separados.jpg")

if __name__ == "__main__":
    src = cv.imread("jit1.JPG", cv.IMREAD_UNCHANGED)
    #Se reescala la imagen para que no se tarde tanto el procesamiento
    #escalada = resize(src)
    src2 = cv.imread("jit1.JPG", cv.IMREAD_UNCHANGED)
    ##cv.imshow('suavizado.jpg',src2)
    if not os.path.exists("gauss.jpg"):
        gauss = cv.GaussianBlur(src2, (17,17), 7)
        cv.imwrite("gauss.jpg",gauss)
    gauss = cv.imread("gauss.jpg")
    print("gauss")
    #Un pequeño inconveniente del kmean es que tengo que buscar la semilla ideal
    #y para cada factor de escala tuve que probar varios datos pero el kmeans funciona
    #escala25  cluster5 seed 200
    #escala50  cluster4 seed 60
    segmentada = procesamientoKmeans(gauss)
    separados = separar_colores(segmentada, np.array([143, 36, 30]))
    # LoG o canny
    if not os.path.exists("canny.jpg"):
        canny = cv.Canny(separados,255,255)
        cv.imwrite("canny.jpg",canny)
    canny = cv.imread("canny.jpg")
    numCarac = 2
    white_pixels_coords = []
    for x in range(canny.shape[0]):
        for y in range(canny.shape[1]):
            if np.array_equal(canny[x,y],np.array([255,255,255])):
                white_pixels_coords.append((x,y))

    Ajustar(np.array(white_pixels_coords))

    sortedc = []
    for cluster in clusters:
        sortedc.append(sorted(cluster, key=lambda x: x[0]))
    
    distancias = src2.copy()

    for cluster in sortedc:
        print(f"Min: {cluster[0]}")
        print(f"Max: {cluster[-1]}")
        distancias = cv.line(
            distancias, (cluster[0][1], cluster[0][0]), (cluster[-1][1], cluster[-1][0]), (0, 255, 255), 2)
    
    
    cv.imwrite("final.jpg",distancias)
    cv.imshow("Distancias",distancias)
    cv.waitKey()

