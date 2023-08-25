import numpy as np
import csv
import scipy.io as spio
from matplotlib import pyplot as p
import meshio
from pyntcloud.geometry.models.plane import Plane
from pyntcloud import PyntCloud
from sklearn.cluster import DBSCAN
from PIL import Image 
import os
from scipy.optimize import minimize
from scipy.spatial import KDTree

# in the functions, the name outReference and orderedReference is used for scanned reference points
# the name outOptodes and orderedOptodes is used for scanned optode points
# montageRefPoints is used for reference points from the montage
# montageOptodes is used for optode points from the montage

#functions needed for preprocessing
def parse_v(fields):
    # fields: ['v', '109.113800', '108.126198', '390.562408']
    return np.asarray([float(i) for i in fields[1:]]) 

def parse_vt(fields):
    # fields: ['vt', '0.736044', '0.825012']
    return np.asarray([float(i) for i in fields[1:]]) 

def parse_f(fields):
    #fields: ['f', '801/38604', '953/38756', '954/38757']

    result = np.zeros((3,2), dtype=int)
    for i in range(3):
        idx_node, idx_uv = fields[1+i].split("/")
        result[i,0] = int(idx_node) - 1 
        result[i,1] = int(idx_uv) - 1

    return result

def preProcessing(instring):
    '''
    #to process the scans, you need an .obj file and a .jpg file
    input: 
        instring: file name without extension

    output: 
        vnew: vertices (N, 3)
        vcolors_rgb: vertex colors in rgb (N, 3)
        vcolors_hsv: vertex colors in hsv (N, 3)

    example:
        vnew, vcolors_rgb, vcolors_hsv=preProcessing("scan0/scan0post")
    '''
    # open file, read and split lines
    with open(instring+".obj") as fin:
        lines = fin.readlines()
        fields = [l.strip().split() for l in lines]

    # vertices, shape (n_vertices, 3)
    v = np.vstack([parse_v(f) for f in fields if f[0] == "v"])

    # vertex textures,  shape (n_uvpositions, 2)
    vt = np.vstack([parse_vt(f) for f in fields if f[0] == "vt"])

    img = p.imread(instring+".jpg")
    img_hsv = np.asarray(Image.fromarray(img).convert("HSV"))

    rows = np.round((1-vt[:,1]) * img.shape[0]).astype(int) 
    rows[rows>=img.shape[0]] = img.shape[0]-1
    cols = np.round(vt[:,0] * img.shape[1]).astype(int)

    # rgb color tuples (0..1.0) for each vertex texture entry
    vtcolors_rgb = img[rows, cols,:].astype(float)/255
    vtcolors_hsv = img_hsv[rows, cols, :].astype(float)/255

    # faces, shape (3*n_meshnodes, 2), maps vertex indices to vertex texture indices
    f = np.vstack([parse_f(f) for f in fields if f[0] == "f"])

    # remove duplicates, this can still contain multiple entries per vertex, if a vertex is mapped to several vertex textures
    # checked a few. Often these different vertex texture entries have the same uv coordniates
    f = np.unique(f, axis=0)

    # arrays that store for each vertex a color in rgb and hsv color space
    vcolors_rgb = np.zeros((len(v), 3), dtype=float)
    vcolors_hsv = np.zeros((len(v), 3), dtype=float)

    for idx_v, idx_vt in f: 
        rgb = vtcolors_rgb[idx_vt]
        hsv = vtcolors_hsv[idx_vt]

        vcolors_rgb[idx_v,:] = rgb
        vcolors_hsv[idx_v,:] = hsv

    #subtract mean to center around origin
    vnew=np.copy(v)
    vnew[:, 0]=v[:, 0]-np.mean(v[:, 0])
    vnew[:, 1]=v[:, 1]-np.mean(v[:, 1])
    vnew[:, 2]=v[:, 2]-np.mean(v[:, 2])

    return vnew, vcolors_rgb, vcolors_hsv

def pointsiInMask(vnew, mask):
    '''
    #returns points in mask
    input:
        vnew: vertices (N, 3)
        mask: mask (N, 1)
    output: 
        newPoints: vertices with mask input 1 (N, 3)
    '''
    newPoints=np.zeros((sum(mask), 3))
    step=0
    size=int(np.shape(vnew)[0])

    for i in range(size):
        if mask[i]:
            newPoints[step, :]=vnew[i, :]
            step+=1
    
    return newPoints

def finalClusters(newPoints, eps, minSamples, numEl):
    '''
    input: 
        newPoints: vertices with mask input 1 (N, 3)
        eps: radius of the neighborhood [mm] (int)
        minSamples: minimum number of points to form a cluster (int)
        numEl: number of points (scanned reference points or optode points) (int)
    output:
        finalClusters: clusters (list of lists:number of lists=numEl; each with size (N, 3), N is number of detected points in the cluster)

    '''
    #initialize model
    dbscan = DBSCAN(eps=eps, min_samples=minSamples) 
    # Fit the filtered points array to the DBSCAN model
    clusters = dbscan.fit_predict(newPoints)
    uniqueLabels = np.unique(clusters)# retunrs -1, 0, 1, ... as many cluster as I want
    clusterPoints = []

    for label in uniqueLabels:
        # if label is -1, it means that the point doesn't belong to any cluster
        if label != -1:
            clusterPoints.append(newPoints[clusters == label])
    #numEl=number of clusters I want
    sortedClusters = sorted(clusterPoints, key=lambda x: len(x), reverse=True)
    #output only the first numEl clusters, with most points detected
    finalClusters = sortedClusters[:numEl]
    return finalClusters

def writePoints(finalClusters, writeFile):
    '''
    input:
        finalClusters: clusters (list of lists:number of lists=numEl; each with size (N, 3), N is number of detected points in the cluster)
        writeFile: name of the file to write the cluster points in (string with {} to be replaced by the number of the cluster)
    no direct output:
        writes the points in the file (as many cluster as there are in finalClusters, each with size (N, 3))
    '''
    #write points in ply file
    for k in range(len(finalClusters)):
        meshio.write(writeFile.format(k+1), mesh=meshio.Mesh(points=finalClusters[k], cells = []), binary=False)

#functions used in getPlane
def objectiveFunction(center, *args):
    #function to minimize, when searching for the center of the circle
    points = args[0]
    d=args[1]
    return np.sum((np.linalg.norm(points - center, axis=1) - d/2)**2) #distance to center-radis

def fitCircleToPoints(points, d):
    #fit circle to clusters
    initial_guess = np.mean(points, axis=0)  # Use the mean of points as initial guess for the circle center
    result = minimize(objectiveFunction, initial_guess, args=(points, d), method='Nelder-Mead')
    center = result.x
    return center

def getPlane(finalClusters, myPath, numEl, readFile, writeFile, diameter):
    '''    
    #fit plane to each cluster
    input:
        finalClusters: clusters (list of lists:number of lists=numEl; each with size (N, 3), N is number of detected points in the cluster)
        myPath: path to the folder where the clusters files are (pathlib.WindowsPath)
        numEl: number of points (scanned reference points or optode points) (int)
        readFile: name of the file to read the cluster points from (string with {} to be replaced by the number of the cluster)
        writeFile: name of the file to write in the normals of fitted planes (string with {} to be replaced by the number of the cluster)
        diameter: diameter of the circle to fit to the cluster [mm] (int)
    no direct output:   
        written files with the normals of the fitted planes (numEl different files with 2 lists: normal (1, 4) and center of the plane (1, 3))
    '''
    #get center of each cluster
    center=np.zeros((len(finalClusters) , 3))
    for j in range(len(finalClusters)):
        center[j, :] = fitCircleToPoints(finalClusters[j], diameter)

    normald=np.zeros((numEl, 4))
    centerProj=np.zeros((numEl, 3))
    for i in range(numEl):
        #get points in cluster
        fullPath=os.path.join(myPath, readFile.format(i+1))

        #get plane normal
        cloud = PyntCloud.from_file(fullPath)
        is_plane = cloud.add_scalar_field("plane_fit", max_dist=2)
        plane = Plane()
        plane.from_point_cloud(cloud.xyz)
        normald[i, :]=plane.get_equation()
        #project center on plane
        scdist=np.dot(center[i, :], normald[i, :3])+normald[i, 3]
        centerProj[i, :]=center[i, :]-scdist*normald[i, :3]
        #write normals and center points
        f=open(writeFile.format(i+1), 'w', newline='')
        writer = csv.writer(f)
        writer.writerow((normald[i, :], centerProj[i, :]))
        f.close()

def writePointsNearPlane(numEl, readFile, readFile2, myPath, distance):
    '''
    #rewrite files containing points in clusters, within distance of plane (cut outliers)
    input: 
        numEl: number of points (scanned reference points or optode points) (int)
        readFile: name of the file to read the cluster points from (string with {} to be replaced by the number of the cluster)
        readFile2: name of the file to read the normals and center of the plane (string with {} to be replaced by the number of the cluster)
        myPath: path to the folder where the clusters files are (pathlib.WindowsPath)
        distance: distance from plane to keep points [mm] (int)
    no direct output:
        written files with the points in the clusters (numEl different files with size (N, 3), N is number of detected points in the cluster)
    '''
    closePoints=np.empty((numEl, 0)).tolist()
    dist = np.empty((numEl, 0)).tolist()
    points = np.empty((numEl, 0)).tolist()
    normal=np.zeros((numEl, 4))

    for num in range(numEl):
        #get points in cluster
        fullPath=os.path.join(myPath,readFile.format(num+1))
        cloud = PyntCloud.from_file(fullPath)
        points[num]=cloud.points.values
        #get plane normal
        data = (readFile2.format(num+1))
        with open(data, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            #get normal of plane
            for row in datareader:
                normald, center=row
        normal[num, :]=np.array(normald[1:-1].split(), dtype=float)
        #get distance of each point from plane
        for i in range(np.shape(points[num])[0]):
            dist[num].append(np.abs(normal[num, 0]*points[num][i, 0]+normal[num, 1]*points[num][i, 1]+normal[num, 2]*points[num][i, 2]+normal[num, 3])/np.sqrt(normal[num, 0]**2+normal[num, 1]**2+normal[num, 2]**2))
        #if distance of point<distance(input) include the point
        for i in range(np.shape(points[num])[0]):
            if dist[num][i]<distance:
                closePoints[num].append(points[num][i, :])
        #rewrite file
        meshio.write(readFile.format(num+1), mesh=meshio.Mesh(points=closePoints[num], cells = []), binary=False)

def subtractOptode(readFile, numEl, sizeOpt):
    '''
    #subtract optode size from each cluster to get point on head surface
    input: 
        readFile: name of the file to read the plane normals from (string with {} to be replaced by the number of the cluster)
        numEl: number of points (scanned reference points or optode points) (int)
        sizeOpt: size of the optode to subtract from the cluster center in direction towards the head [mm] (int)
    output:
        out: array with the points on the head surface (numEl, 3)
    '''
    out=np.zeros((numEl, 3))
    #get normal and center of plane
    for num in range(numEl):
        data = (readFile.format(num+1))
        with open(data, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                normal, centerProj=row
        normal=np.array(normal[1:-1].split(), dtype=float)
        centerProj=np.array(centerProj[1:-1].split(), dtype=float)
        #depending on the position of the plane, reorient the vector
        if normal[3]>0:
            v1=np.array([-normal[0], -normal[1], -normal[2]])
        else:
            v1=np.array([normal[0], normal[1], normal[2]])
        #subtract optode size from center of cluster
        out[num, :]=centerProj-v1*sizeOpt
    return out

def orderReferencePoints6(numScannedReference, outReference, twoPoints):
    '''
    #for this function to work, 6 reference stickers are needed
    #additional sticker is placed next to either "Nz" or "Iz"; it needs to be further away from Cz as "Nz" or "Iz";
    #the sticker needs to be approximately on the same plane with Cz, Iz and Nz 
    #the closest points are Nz or Iz and the sticker next to it, dependes on where you put two stickers together
    #opposite is the other one Iz or Nz
    #Cz is the point closest to the plane, defined by the 3 previous points
    #of the 2 points closest together, Nz or Iz is the one closest to Cz (we have to put the stickers on the cap that way)
    #Determining Rpa and Lra from cross product of known points

    input:
        numScannedReference: number of scanned reference points (int)=6
        outReference: array with the scanned reference points (numEl, 3)
        two points: label where there are two stickers together (string "Nz" or "Iz")
    output:
        orderedReference: array with the ordered scanned reference points (numEl, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        order: order of the scanned reference points in outReference (1, 5) #example: [4, 0, 2, 3, 5]->[Nz, Iz, Rpa, Lpa, Cz] Nz is in the 4th line in outReference...
    '''
    #find 2 closest points
    distRef=np.zeros((numScannedReference, numScannedReference))
    for i in range(numScannedReference):
        step=0
        for j in range(numScannedReference):
            distRef[i, j]=np.sqrt((outReference[i, 0]-outReference[j, 0])**2+(outReference[i, 1]-outReference[j, 1])**2+(outReference[i, 2]-outReference[j, 2])**2)
    close1=np.where(distRef==np.min(distRef[np.nonzero(distRef[:, :])]))[0][0]
    close2=np.where(distRef==np.min(distRef[np.nonzero(distRef[:, :])]))[0][1]

    #find opposite point
    idx0=np.zeros(4)
    dist2=np.zeros(4)
    step=0
    for i in range(numScannedReference):
        #use only undetermined points
        if i!=close1 and i!=close2:
            idx0[step]=i
            dist2[step]=np.sqrt((outReference[i, 0]-outReference[close1, 0])**2+(outReference[i, 1]-outReference[close1, 1])**2+(outReference[i, 2]-outReference[close1, 2])**2)
            step=step+1
    opposite=int(idx0[np.argmax(dist2)])

    #Define plane with close1, 2, opposite -> Cz is the point closest to this plane
    v1 = outReference[close1]-outReference[opposite]
    v2 = outReference[close2]-outReference[opposite]
    cp = np.cross(v1, v2)
    d = np.dot(cp, outReference[close1])
    idx1=np.zeros(3)
    distance2Plane=np.zeros((3))
    step=0
    for i in range(numScannedReference):
        #use only undetermined points
        if i!=close1 and i!= close2 and i!=opposite:
            idx1[step]=i
            distance2Plane[step]=np.abs(cp[0]*outReference[i, 0]+cp[1]*outReference[i, 1]+cp[2]*outReference[i, 2]+d)/np.sqrt(cp[0]**2+cp[1]**2+cp[2]**2)
            step+=1
    Cz=int(idx1[np.argmin(distance2Plane)])

    #Nz or Iz is the one of Nz1 and Nz2 closest to Cz, Iz or Nz is opposite
    if twoPoints=="Nz":
        distNz=np.zeros(2)
        distNz[0]=np.sqrt((outReference[Cz, 0]-outReference[close1, 0])**2+(outReference[Cz, 1]-outReference[close1, 1])**2+(outReference[Cz, 2]-outReference[close1, 2])**2)
        distNz[1]=np.sqrt((outReference[Cz, 0]-outReference[close2, 0])**2+(outReference[Cz, 1]-outReference[close2, 1])**2+(outReference[Cz, 2]-outReference[close2, 2])**2)
        if np.argmin(distNz)==0:
            Nz=close1
        else:
            Nz=close2
        Iz=opposite
    elif twoPoints=="Iz":
        distIz=np.zeros(2)
        distIz[0]=np.sqrt((outReference[Cz, 0]-outReference[close1, 0])**2+(outReference[Cz, 1]-outReference[close1, 1])**2+(outReference[Cz, 2]-outReference[close1, 2])**2)
        distIz[1]=np.sqrt((outReference[Cz, 0]-outReference[close2, 0])**2+(outReference[Cz, 1]-outReference[close2, 1])**2+(outReference[Cz, 2]-outReference[close2, 2])**2)
        if np.argmin(distIz)==0:
            Iz=close1
        else:
            Iz=close2
        Nz=opposite

    #Determining Rpa and Lra from cross product of known points
    idx2=np.zeros(2)
    step=0
    for i in range(numScannedReference):
        #use only undetermined points
        if i!=close1 and i!= close2 and i!=opposite and i!=Cz:
            idx2[step]=i
            step+=1
    cr=np.cross(outReference[Nz, :]-outReference[Cz, :], outReference[Iz, :]-outReference[Cz, :])
    cr=cr/np.linalg.norm(cr)
    #if the result of the dot product is positive, the first point is Lpa, otherwise it is Rpa
    if np.dot(cr, outReference[int(idx2[0]), :])>0:
        Lpa=int(idx2[0])
        Rpa=int(idx2[1])
    else:
        Lpa=int(idx2[1])
        Rpa=int(idx2[0])

    orderedReference=np.zeros((5, 3))
    orderedReference[0, :]=outReference[Nz, :]
    orderedReference[1, :]=outReference[Iz, :]
    orderedReference[2, :]=outReference[Rpa, :]
    orderedReference[3, :]=outReference[Lpa, :]
    orderedReference[4, :]=outReference[Cz, :]

    return  orderedReference, np.array([Nz, Iz, Rpa, Lpa, Cz])

#used for 5 scanned reference points, doesn't work for all montages;
#not used in the script
def orderReferencePoints(numScannedOptodes, numScannedReference, outReference, outOptodes):
    '''
    #for this function to work, 5 reference stickers are needed;
    #it works only if the mean distance to the 5 closest scanned optode points from every reference point is smallest for Iz
    #Cz is the point with lowest std of distances to other scanned reference points
    #Iz is the point with closest mean distance to 5 closest scanned optode points
    #Nz is the point with largest distance to Iz
    #Determining Rpa and Lra from cross product of known points

    input:
        numScannedOptodes: number of scanned optodes points
        numScannedReference: number of scanned reference points=5
        outReference: scanned reference points (5, 3)
        outOptodes: scanned optode points (numScannedOptodes, 3)
    output:
        orderedReference: ordered points from scanned reference points (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
    '''
    #Cz is the point with lowest std of distances to other scanned reference points
    distRef=np.zeros((numScannedReference, numScannedReference))
    stds=np.zeros(numScannedReference)
    arr=np.zeros(4)
    for i in range(numScannedReference):
        step=0
        for j in range(numScannedReference):
            distRef[i, j]=np.sqrt((outReference[i, 0]-outReference[j, 0])**2+(outReference[i, 1]-outReference[j, 1])**2+(outReference[i, 2]-outReference[j, 2])**2)
            if i!=j:
                arr[step]=distRef[i, j]
                step+=1

        stds[i]=np.std(arr[:])
    Cz=np.argmin(stds)

    #Iz is the point with closest mean distance to 5 closest scanned optodes points
    idx0=np.zeros(4)
    step=0
    distOther=np.zeros((4, numScannedOptodes))
    means=np.zeros(4)
    for i in range(numScannedReference):
        #use only undetermined points
        if i!=Cz:
            idx0[step]=i
            #calculate distance to all scanned optodes points
            for j in range(numScannedOptodes):
                distOther[step, j]=np.sqrt((outReference[i, 0]-outOptodes[j, 0])**2+(outReference[i, 1]-outOptodes[j, 1])**2+(outReference[i, 2]-outOptodes[j, 2])**2)
            distOther[step, :]=np.sort(distOther[step, :])
            #calculate mean distance to 5 closest scanned optodes points
            means[step]=np.mean(distOther[step, :5])
            step+=1
    Iz=int(idx0[np.argmin(means)])

    #Nz is the point with largest distance to Iz
    idx1=np.zeros(3)
    step=0
    distRef2=np.zeros(3)
    for i in range(numScannedReference):
        #use only undetermined points
        if i!=Iz and i!=Cz:
            idx1[step]=i
            #distance to Iz
            distRef2[step]=np.sqrt((outReference[i, 0]-outReference[Iz, 0])**2+(outReference[i, 1]-outReference[Iz, 1])**2+(outReference[i, 2]-outReference[Iz, 2])**2)
            step+=1
    Nz=int(idx1[np.argmax(distRef2)])

    #Determining Rpa and Lra from cross product of known points
    idx2=np.zeros(2)
    step=0
    for i in range(numScannedReference):
        #use only undetermined points
        if i!=Nz and i!=Iz and i!=Cz:
            idx2[step]=i
            step+=1
    cr=np.cross(outReference[Nz, :]-outReference[Cz, :], outReference[Iz, :]-outReference[Cz, :])
    cr=cr/np.linalg.norm(cr)
    #if the result of the dot product is positive, the first point is Lpa, otherwise it is Rpa
    if np.dot(cr, outReference[int(idx2[0]), :])>0:
        Lpa=int(idx2[0])
        Rpa=int(idx2[1])
    else:
        Lpa=int(idx2[1])
        Rpa=int(idx2[0])

    orderedReference=np.zeros((5, 3))
    orderedReference[0, :]=outReference[Nz, :]
    orderedReference[1, :]=outReference[Iz, :]
    orderedReference[2, :]=outReference[Rpa, :]
    orderedReference[3, :]=outReference[Lpa, :]
    orderedReference[4, :]=outReference[Cz, :]

    return  orderedReference, np.array([Nz, Iz, Rpa, Lpa, Cz])
    
def writePointsTxt(numSor, numDet, orderedReference, outOptodes, readFile, writeFile):
    '''
    input:
        numSor: number of sources (int)
        numDet: number of detectors (int)
        orderedReference: ordered scanned reference points (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        outOptodes: scanned optodes points  (numScannedOptodes, 3)
        readFile: name of the file to read the coordinates to align to = Standard_Optodes.txt file from the used montage; montage reference points need to be labeled O01-O05 in order Nz, Iz, Rpa, Lpa, Cz (string)
        filename: name of the file to write to (string)
    output:
        orderedOptodes:  ordered scanned optodes points (numScannedOptodes, 3)->order is determined by points in Standard_Optodes.txt (readFile input)
        orderedOutReference: ordered scanned reference points (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        errLabels: error of each point to closest point in readFile input(1, numScannedOptodes)->order is determined by points in readFile input (Standard_Optodes.txt)

    example: 
        orderedOptodes, orderedReference, errLabels=writePointsTxt(31, 30, orderedReference, outOptodes, "Standard_Optodes.txt", "writeFile.txt")
    '''
    #read .txt file
    content=np.loadtxt(readFile, dtype='str', delimiter=',')
    #montage reference labels in input file
    labelsO=["O01", "O02", "O03", "O04", "O05"]
    #define montage reference and montage optode points
    montageRefPoints=np.zeros((5, 3))
    for i in range(5):
        montageRefPoints[i, :]=np.array(content[np.where(content[:, 0]==labelsO[i]), 1:], dtype='float')
    numScannedOptodes=numSor+numDet
    montageOptodes=np.zeros((numScannedOptodes, 3))
    for i in range(numScannedOptodes):
        montageOptodes[i, :]=np.array(content[i, 1:], dtype='float')
    #define labels
    labelsRef=["Nz", "Iz", "Rpa", "Lpa", "Cz"]

    #get affine transformation from scanned reference points to montage reference points
    trn = affineFit(montageRefPoints, orderedReference)
    #use transformations to transform montageOptodes
    trMontageOptodes=trn.Transform(montageOptodes.T)
    trMontageOptodes=np.array(trMontageOptodes).T
    #use icp for better alignment so the corresponding points are found
    trMontageOptodes2, _ = icp(outOptodes, trMontageOptodes)
    #calculate error to find the correspoinding points between outOptodes (scanned optodes) and trMontageOptodes (transformed montage optodes)
    err=np.zeros((numScannedOptodes, numScannedOptodes))
    err2=np.zeros((numScannedOptodes, 2))
    for i in range(numScannedOptodes):
        for j in range(numScannedOptodes):
            err[i, j]=np.sqrt((outOptodes[i, 0]-trMontageOptodes2[j, 0])**2+(outOptodes[i, 1]-trMontageOptodes2[j, 1])**2+(outOptodes[i, 2]-trMontageOptodes2[j, 2])**2)
        err2[i, 0]=min(err[i, :])
        err2[i, 1]=np.argmin(err[i, :])

    #check if each point is closest to a different point
    if len(err2[:, 1]) != len(set(err2[:, 1])):
        print('Error: some points are closest to the same point')
        print(len(err2[:, 1]) , len(set(err2[:, 1])))
    orderedOptodes=np.zeros_like(outOptodes)
    #same order of points as Standard_Optodes.txt
    errLabels=np.zeros(numScannedOptodes)
    for i in range(numScannedOptodes):
        orderedOptodes[i, :]=outOptodes[np.where(err2[:, 1]==i), :]
        errLabels[i]=err2[np.where(err2[:, 1]==i), 0]
    #write in txt file
    with open(writeFile, 'w') as f:
        for i in range(5):
            f.write(labelsRef[i]+',{}, {}, {}'.format(orderedReference[i, 0], orderedReference[i, 1], orderedReference[i, 2]))
            f.write('\n')
        for i in range(numScannedOptodes):
            f.write(content[i, 0]+',{}, {}, {}'.format(orderedOptodes[i, 0], orderedOptodes[i, 1], orderedOptodes[i, 2]))
            f.write('\n')
    return orderedOptodes, orderedReference, errLabels

def processOne(readFileScan, readFileOptodes, readFileClustersReference, readFileClustersOptode, readFilePlaneReference, readFilePlaneOptode, writeFile, masks, numSor, numDet, numScannedOptodes, numScannedReference, 
               radiusOptodes, radiusReference, myPath, subtractOptodeSize, subtractReferenceSize, distance2Plane, twoPoints, printErr=False, plot=True):
    '''
    input:
        readFileScan: scan file (string)
        readFileOptodes: optodes file Standard_Optodes.txt (string)
        readFileClustersReference: clusters file for scanned reference points (string)
        readFileClustersOptode: clusters file for scanned optodes (string)
        readFilePlaneReference: plane file for scanned reference points (string)
        readFilePlaneOptode: plane file for scanned optodes(string)
        writeFile: name of the file to write to (string)
        masks: masks for stickers, used as reference points and on top of optodes (list of 12 floats between 0 and 1)
        numSor: number of sources (int)
        numDet: number of detectors (int)
        numScannedOptodes: number of scanned optodes points (int)
        numScannedReference: number of scanned reference points (int)
        radiusOptodes: size of the circle to fit to the optode clusters [mm] (float)
        radiusReference: size of the circle fit to the reference clusters [mm] (float)
        myPath: path to the folder containing the files (WindowsPath)
        subtractOptodeSize: size to subtract from the optode cluster to get wanted point on the scalp [mm] (float)
        subtractReferenceSize: size to subtract from the reference cluster to get wanted point [mm] (float)
        distance2Plane: distance from fitted plane to keep points in clusters [mm] (float)
        two points: label where there are two stickers together (string "Nz" or "Iz")
        printErr: print mean and std of distance from aligned points to points in Standard_Optodes.txt [mm] (bool)
        plot: plot the alignment (bool)

    output:
        orderedOptodes: ordered scanned optode points - same order as in the input readFileOptodes (Standard_Optodes.txt) (numScannedOptodes, 3)
        orderedReference: ordered scanned reference points (numScannedReference, 3)
        errLabels: error with same order as in the input readFileOptodes (Standard_Optodes.txt) (1, numScannedOptodes)

    example:
        orderedOptodes, orderedReference, errLabels=f.processAll("scan/scan2post"", "Standard_Optodes.txt", "scan/pointsReference{num}.ply".format(num={}), "scan/pointsOptode{num}.ply".format(num={}), 
        "scan/normals(+d)Reference{num}.csv".format(num={}), "scan/normals(+d)Optode{num}.csv".format(num={}), "coordsTest.txt", [0.14 0.035, 0.65, 0.35, 0.8, 0.3, 0.35, 0.1, 0.45, 0.35, 0.45, 0.15], 31, 30, 61, 6, 
        6.5, 5, pathlib.Path().resolve(), 22.6, 0, 2, "Iz", printErr=True, plot=False)
    '''
    
    yHueCenter=masks[0]
    yHueWidth=masks[1]
    ySatCenter=masks[2]
    ySatWidth=masks[3]
    yValueCenter=masks[4]
    yValueWidth=masks[5]

    bHueCenter=masks[6]
    bHueWidth=masks[7]
    bSatCenter=masks[8]
    bSatWidth=masks[9]
    bValueCenter=masks[10]
    bValueWidth=masks[11]

    #get vertex locations and colors
    vnew, vcolors_rgb, vcolors_hsv=preProcessing(readFileScan)
    #define mask
    MaskOptodes=(np.abs(vcolors_hsv[:,0] - yHueCenter) < yHueWidth) & (np.abs(vcolors_hsv[:,1] - ySatCenter) < ySatWidth) & (np.abs(vcolors_hsv[:,2] - yValueCenter) < yValueWidth)
    MaskReference=(np.abs(vcolors_hsv[:,0] - bHueCenter) < bHueWidth) & (np.abs(vcolors_hsv[:,1] - bSatCenter) < bSatWidth) & (np.abs(vcolors_hsv[:,2] - bValueCenter) < bValueWidth)
    #get points in mask
    newPointsOptodes=pointsiInMask(vnew, MaskOptodes)
    newPointsReference=pointsiInMask(vnew, MaskReference)
    #get as many clusters as there are stickers
    finalClustersOptodes=finalClusters(newPointsOptodes, eps=radiusOptodes, minSamples=50, numEl=numScannedOptodes)
    finalClustersReference=finalClusters(newPointsReference, eps=radiusReference, minSamples=50, numEl=numScannedReference)
    #write points belonging to each cluster to a file
    writePoints(finalClustersOptodes, readFileClustersOptode)
    writePoints(finalClustersReference, readFileClustersReference)
    #fit a plane to each cluster
    getPlane(finalClustersOptodes, myPath, numScannedOptodes, readFileClustersOptode, readFilePlaneOptode, diameter=2*radiusOptodes)
    getPlane(finalClustersReference, myPath, numScannedReference, readFileClustersReference, readFilePlaneReference, diameter=2*radiusOptodes)
    #cut points too far from the plane and rewrite the points file
    writePointsNearPlane(numScannedOptodes, readFileClustersOptode, readFilePlaneOptode, myPath, distance2Plane)
    writePointsNearPlane(numScannedReference, readFileClustersReference, readFilePlaneReference, myPath, distance2Plane)
    #re-fit the plane
    getPlane(finalClustersOptodes, myPath, numScannedOptodes, readFileClustersOptode, readFilePlaneOptode, diameter=2*radiusOptodes)
    getPlane(finalClustersReference, myPath, numScannedReference, readFileClustersReference, readFilePlaneReference,  diameter=2*radiusOptodes)
    #subtract optode size from each point to get a point on the surface of the head
    outOptodes=subtractOptode(readFilePlaneOptode, numScannedOptodes, subtractOptodeSize)
    #subtractReferenceSize is set to 0, can be modified if needed
    outReference=subtractOptode(readFilePlaneReference, numScannedReference, subtractReferenceSize)
    #order the scanned reference points, use a different function for 6 reference stickers
    #5 scanned reference points function works for montages, where Iz has the lowest mean distance to closest 5 scanned optode points
    if numScannedReference==5:
        orderedReference, order=orderReferencePoints(numScannedOptodes, numScannedReference, outReference, outOptodes)
    elif numScannedReference==6:
        orderedReference, order=orderReferencePoints6(numScannedReference, outReference, twoPoints)
    #wrtie the final file and return ordered scanned optode points, ordered scanned reference points and error to closest point in input readFileOptodes (Standard_Optodes.txt)
    orderedOptodes, orderedReference, errLabels=writePointsTxt(numSor, numDet, orderedReference, outOptodes, readFileOptodes, writeFile)
    #print mean and std of error
    if printErr==True:
        print(np.mean(errLabels[:]), np.std(errLabels[:]))
    #plot the alignment
    if plot==True:
        #read .txt file
        content=np.loadtxt(readFileOptodes, dtype='str', delimiter=',')
        #montage reference labels in input file
        labelsO=["O01", "O02", "O03", "O04", "O05"]
        #define montage reference and optode points
        montageRefPoints=np.zeros((5, 3))
        for i in range(5):
            montageRefPoints[i, :]=np.array(content[np.where(content[:, 0]==labelsO[i]), 1:], dtype='float')
        numScannedOptodes=numSor+numDet
        montageOptodes=np.zeros((numScannedOptodes, 3))
        for i in range(numScannedOptodes):
            montageOptodes[i, :]=np.array(content[i, 1:], dtype='float')
        #define labels
        labelsRef=["Nz", "Iz", "Rpa", "Lpa", "Cz"]

        #get affine transformation from scanned reference points to montage reference points
        trn = affineFit(montageRefPoints, orderedReference)
        #use transformations to transform montageRefPoints points
        trMontageRefPoints=trn.Transform(montageRefPoints.T)
        trMontageRefPoints=np.array(trMontageRefPoints).T
        #use transformations to transform montageOptodes points
        trMontageOptodes=trn.Transform(montageOptodes.T)
        trMontageOptodes=np.array(trMontageOptodes).T
        #use icp for better alignment so the corresponding points are found
        trMontageOptodes2, _ = icp(outOptodes, trMontageOptodes)
        #plot alignment
        fig = p.figure(figsize=(10, 10)); 
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        fig.suptitle('First alignment before and after affine transformations of the reference points', fontsize=16)
        
        ax.scatter(montageRefPoints[:, 0], montageRefPoints[:, 1], montageRefPoints[:, 2], s=10, color='r', label='unaligned montage reference points')
        ax.scatter(orderedReference[:, 0], orderedReference[:, 1], orderedReference[:, 2], s=10, color='b', label='scanned reference points')
        ax.legend()
        
        ax=fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(trMontageRefPoints[:, 0], trMontageRefPoints[:, 1], trMontageRefPoints[:, 2], s=10, color='g', label='aligned montage reference points')
        ax.scatter(orderedReference[:, 0], orderedReference[:, 1], orderedReference[:, 2], s=10, color='b', label='scanned reference points')
        ax.legend() 

        fig = p.figure(figsize=(10, 10)); 
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        fig.suptitle('Fine alignment before and after icp transformations of the optode points', fontsize=16)
        ax.scatter(trMontageOptodes[:, 0], trMontageOptodes[:, 1], trMontageOptodes[:, 2], s=10, color='r', label='unaligned montage optode points')
        ax.scatter(outOptodes[:, 0], outOptodes[:, 1], outOptodes[:, 2], s=10, color='b', label='scanned optode points')
        ax.legend()

        ax=fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(trMontageOptodes2[:, 0], trMontageOptodes2[:, 1], trMontageOptodes2[:, 2], s=10, color='g', label='aligned montage optode points')
        ax.scatter(outOptodes[:, 0], outOptodes[:, 1], outOptodes[:, 2], s=10, color='b', label='scanned optode points')
        ax.legend()

    return orderedOptodes, orderedReference, errLabels

#other functions used in the script
def icp(reference_points, target_points, max_iterations=15, tolerance=1e-5):
    """
    Perform Iterative Closest Point (ICP) algorithm for point cloud registration.
    used for more accurate alignment of the colin model for labeling 

    input:
        reference_points (np.ndarray): Reference point cloud.
        target_points (np.ndarray): Target point cloud.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    outout:
        np.ndarray: Transformed target points.
        np.ndarray: Final transformation matrix.
    """
    transformed_points = np.copy(target_points)
    transformation_matrix = np.identity(3)  # Initial transformation matrix

    for iteration in range(max_iterations):
        # Find nearest neighbors
        tree = KDTree(reference_points)
        distances, indices = tree.query(transformed_points)

        # Compute transformation matrix using SVD
        matched_reference_points = reference_points[indices]
        centroid_target = np.mean(transformed_points, axis=0)
        centroid_reference = np.mean(matched_reference_points, axis=0)
        H = np.dot((transformed_points - centroid_target).T, matched_reference_points - centroid_reference)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        t = centroid_reference - np.dot(R, centroid_target)

        # Apply transformation to target points
        transformed_points = np.dot(transformed_points, R.T) + t

        # Calculate mean squared error
        mse = np.mean(distances ** 2)
        #print(f"Iteration {iteration + 1}, Mean Squared Error: {mse}")

        # Check for convergence
        if mse < tolerance:
            #print("Converged!")
            break

    return transformed_points, np.hstack((R, t.reshape(-1, 1)))

def rigidTransform3D(A, B):
    '''
    input: 
        A: points to be aligned (N, 3)
        B: reference points (N, 3)
    output:
        R: rotation matrix (3, 3)
        t: translation vector (1, 3)

    example:
        R, t=rigitTransform3D(A, B)
    '''

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def affineFit(fromPts, toPts):
    """
    Fit an affine transformation to given point sets.
    More precisely: solve (least squares fit) matrix 'A'and 't' from
    'p ~= A*q+t', given vectors 'p' and 'q'.
    Works with arbitrary dimensional vectors (2d, 3d, 4d...).
    Written by Jarno Elonen <elonen@iki.fi> in 2007.
    Placed in Public Domain.
    Based on paper "Fitting affine and orthogonal transformations
    between two sets of points, by Helmuth Späth (2003).
    
    input:
        fromPts= point to be transformed (N, 3)
        toPts= reference points (N, 3)
    output:
        transformation matrix (3, 3) to transform fromPts to toPts

    example:
        transformation=affineFit(fromPts, toPts)
    """

    q = fromPts
    p = toPts
    if np.shape(q) != np.shape(p) or np.shape(q)[0]<1 or np.shape(q)[1]<1:
    #if len(q) != len(p) or len(q)<1:
        print("fromPts and toPts must be of same size.")
        return False

    #dim = len(q[0]) # num of dimensions
    dim=np.shape(q)[1]
    if len(q) < dim:
        print("Too few points => under-determined system.")
        return False

    # Make an empty (dim) x (dim+1) matrix and fill it
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]
    for j in range(dim):
        for k in range(dim+1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # Make an empty (dim+1) x (dim+1) matrix and fill it
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim+1):
            for j in range(dim+1):
                Q[i][j] += qt[i] * qt[j]

    # Ultra simple linear system solver. Replace this if you need speed.
    def gauss_jordan(m, eps = 1.0/(10**10)):
      """Puts given matrix (2D array) into the Reduced Row Echelon Form.
         Returns True if successful, False if 'm' is singular.
         NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
         Written by Jarno Elonen in April 2005, released into Public Domain"""
      (h, w) = (len(m), len(m[0]))
      for y in range(0,h):
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
          if abs(m[y2][y]) > abs(m[maxrow][y]):
            maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        if abs(m[y][y]) <= eps:     # Singular?
          return False
        for y2 in range(y+1, h):    # Eliminate column y
          c = m[y2][y] / m[y][y]
          for x in range(y, w):
            m[y2][x] -= m[y][x] * c
      for y in range(h-1, 0-1, -1): # Backsubstitute
        c  = m[y][y]
        for y2 in range(0,y):
          for x in range(w-1, y-1, -1):
            m[y2][x] -=  m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):       # Normalize row y
          m[y][x] /= c
      return True

    # Augement Q with c and solve Q * a' = c by Gauss-Jordan
    M = [ Q[i] + c[i] for i in range(dim+1)]
    if not gauss_jordan(M):
        print("Error: singular matrix. Points are probably coplanar.")
        return False

    # Make a result object
    class Transformation:
        """Result object that represents the transformation
           from affine fitter."""

        def To_Str(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    str +="x%d * %f + " % (i, M[i][j+dim+1])
                str += "%f" % M[dim][j+dim+1]
                res += str + "\n"
            return res

        def Transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j+dim+1]
                res[j] += M[dim][j+dim+1]
            return res
    return Transformation()

