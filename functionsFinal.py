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


#in the variable names and comments, blue points and stickers are used for scanned reference stickers; 
#yellow points and stickers are used for the stickers on the other optodes
#reference points refer to points, used for alignment of the blue points

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
        vnew, vcolors_rgb, vcolors_hsv=preProcessing("scan0/scan1post")
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

        #print(idx_v, idx_vt)
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
        eps: radius of the neighborhood (int)
        minSamples: minimum number of points to form a cluster (int)
        numEl: number of stickers (int)
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
        meshio.write(writeFile.format(k+1), mesh=meshio.Mesh(points=finalClusters[k], cells = []), binary=False)#"scan{}/pointsYellow{}.ply".format(scan, k+1), mesh=meshio.Mesh(points=finalClusters[k], cells = []), binary=False)

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
        numEl: number of stickers (int)
        readFile: name of the file to read the cluster points from (string with {} to be replaced by the number of the cluster)
        writeFile: name of the file to write in the normals of fitted planes (string with {} to be replaced by the number of the cluster)
        diameter: diameter of the circle to fit to the cluster (int)
    no direct output:   
        written files with the normals of the fitted planes (numEl different files with 2 lists: normal (1, 4) and center of the plane (1, 3))
    '''
    #get cennter of each cluster
    center=np.zeros((len(finalClusters) , 3))
    for j in range(len(finalClusters)):
        center[j, :] = fitCircleToPoints(finalClusters[j], diameter)

    normald=np.zeros((numEl, 4))
    centerProj=np.zeros((numEl, 3))
    for i in range(numEl):
        #get points in cluster
        fullPath=os.path.join(myPath, readFile.format(i+1))#"scan{}\pointsYellow{}.ply".format(scan, i+1))

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
        f=open(writeFile.format(i+1), 'w', newline='')#'scan{}/normals(+d)Yellow{}.csv'.format(scan, i+1), 'w', newline='')
        writer = csv.writer(f)
        writer.writerow((normald[i, :], centerProj[i, :]))
        f.close()

def writePointsNearPlane(numEl, readFile, readFile2, myPath, distance):
    '''
    #rewrite files containing points in clusters, within distance of plane (cut outliers)
    input: 
        numEl: number of stickers (int)
        readFile: name of the file to read the cluster points from (string with {} to be replaced by the number of the cluster)
        readFile2: name of the file to read the normals and center of the plane (string with {} to be replaced by the number of the cluster)
        myPath: path to the folder where the clusters files are (pathlib.WindowsPath)
        distance: distance from plane to keep points (int)
    no direct output:
        written files with the points in the clusters (numEl different files with size (N, 3), N is number of detected points in the cluster)
    '''
    closePoints=np.empty((numEl, 0)).tolist()
    dist = np.empty((numEl, 0)).tolist()
    points = np.empty((numEl, 0)).tolist()
    normal=np.zeros((numEl, 4))

    for num in range(numEl):
        #get points in cluster
        fullPath=os.path.join(myPath,readFile.format(num+1))#"scan{}\pointsYellow{}.ply".format(scan, num+1))
        cloud = PyntCloud.from_file(fullPath)
        points[num]=cloud.points.values
        #get plane normal
        data = (readFile2.format(num+1))#('scan{}/normals(+d)Yellow{}.csv'.format(scan, num+1))
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
        meshio.write(readFile.format(num+1), mesh=meshio.Mesh(points=closePoints[num], cells = []), binary=False)#"scan{}/pointsBlue{}.ply".format(scan, num+1), mesh=meshio.Mesh(points=closePoints[num], cells = []), binary=False)


def subtractOptode(readFile, numEl, sizeOpt):
    '''
    #subtract optode size from each cluster to get point on head surface
    input: 
        readFile: name of the file to read the plane normals from (string with {} to be replaced by the number of the cluster)
        numEl: number of blue stickers (int)
        sizeOpt: size of the optode to subtract from the sticker center in direction towards the head [mm] (int)
    output:
        out: array with the points on the head surface (numEl, 3)
    '''
    out=np.zeros((numEl, 3))
    #get normal and center of plane
    for num in range(numEl):
        data = (readFile.format(num+1))#('scan{}/normals(+d)Yellow{}.csv'.format(scan, num+1))
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

def subtractCapThickness(outBlue, numEl, readFile,  order, capThickness, subtract=["Cz", "Iz", "Rpa", "Lpa"]):
    '''
    #subtract sticekrs where are placed on the cap(usually Cz, Iz, Rpa and Lpa)
    input:
        outBlue: array with the points on the head surface (numEl, 3)
        numEl: number of stickers (int)
        readFile: name of the file to read the plane normals from (string with {} to be replaced by the number of the cluster)
        order: order of the blue points in outBlue (5, 3) #example: [4, 0, 2, 3, 1]->[Nz, Iz, Rpa, Lpa, Cz] Nz is in the 4th line in outBlue...
        capThickness: thickness of the cap [mm] (int)
        subtract: list of labels of the stickers to subtract (list of strings)
    output:
        outBlueNew: array with the points including subtracted cap thickness (numEl, 3)

    '''
    allLabelsBlue=["Nz", "Iz", "Rpa", "Lpa", "Cz"]
    outBlueNew=np.zeros((numEl, 3))
    #get normal of plane and cetner of cluster
    for num, label in enumerate(allLabelsBlue):
        if label in subtract:
            data = (readFile.format(num+1))#('scan{}/normals(+d)Yellow{}.csv'.format(scan, num+1))
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
            #subtract cap thickness from center of cluster
            outBlueNew[num, :]=centerProj-v1*capThickness
        else:
            outBlueNew[num, :]=outBlue[order[num], :]
    return outBlueNew

def orderReferencePoints6(numElBlue, outBlue, twoPoints):
    '''
    #for this function to work, 6 blue stickers are needed
    #additional sticker is placed next to either "Nz" or "Iz"; it needs to be further away from Cz as "Nz" or "Iz";
    #the sticker needs to be approximately on the same plane with Cz, Iz and Nz 
    #closest points are Nz or Iz and the sticker next to it, dependes on where you put two stickers together
    #opposite is the other one Iz or Nz
    #Cz is the point closest to the plane, defined by the 3 previous points
    #of the 2 points closest together, Nz or Iz is the one closest to Cz (we have to put the stickers on the cap that way)
    #Determining Rpa and Lra from cross product of known points

    input:
        numElBlue: number of stickers (int)=6
        outBlue: array with the blue points (numEl, 3)
        two points: label where there are two stickers together (string "Nz" or "Iz")
    output:
        orderedBlue: array with the ordered blue points (numEl, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        order: order of the blue points in outBlue (1, 5) #example: [4, 0, 2, 3, 5]->[Nz, Iz, Rpa, Lpa, Cz] Nz is in the 4th line in outBlue...
    '''
    #find 2 closest points
    distRef=np.zeros((numElBlue, numElBlue))
    for i in range(numElBlue):
        step=0
        for j in range(numElBlue):
            distRef[i, j]=np.sqrt((outBlue[i, 0]-outBlue[j, 0])**2+(outBlue[i, 1]-outBlue[j, 1])**2+(outBlue[i, 2]-outBlue[j, 2])**2)
    close1=np.where(distRef==np.min(distRef[np.nonzero(distRef[:, :])]))[0][0]
    close2=np.where(distRef==np.min(distRef[np.nonzero(distRef[:, :])]))[0][1]

    #find opposite point
    idx0=np.zeros(4)
    dist2=np.zeros(4)
    step=0
    for i in range(numElBlue):
        #use only undetermined points
        if i!=close1 and i!=close2:
            idx0[step]=i
            dist2[step]=np.sqrt((outBlue[i, 0]-outBlue[close1, 0])**2+(outBlue[i, 1]-outBlue[close1, 1])**2+(outBlue[i, 2]-outBlue[close1, 2])**2)
            step=step+1
    opposite=int(idx0[np.argmax(dist2)])

    #Define plane with close1, 2, opposite -> Cz is the point closest to this plane
    v1 = outBlue[close1]-outBlue[opposite]
    v2 = outBlue[close2]-outBlue[opposite]
    cp = np.cross(v1, v2)
    d = np.dot(cp, outBlue[close1])
    idx1=np.zeros(3)
    distance2Plane=np.zeros((3))
    step=0
    for i in range(numElBlue):
        #use only undetermined points
        if i!=close1 and i!= close2 and i!=opposite:
            idx1[step]=i
            distance2Plane[step]=np.abs(cp[0]*outBlue[i, 0]+cp[1]*outBlue[i, 1]+cp[2]*outBlue[i, 2]+d)/np.sqrt(cp[0]**2+cp[1]**2+cp[2]**2)
            step+=1
    Cz=int(idx1[np.argmin(distance2Plane)])

    #Nz or Iz is the one of Nz1 and Nz2 closest to Cz, Iz or Nz is opposite
    if twoPoints=="Nz":
        distNz=np.zeros(2)
        distNz[0]=np.sqrt((outBlue[Cz, 0]-outBlue[close1, 0])**2+(outBlue[Cz, 1]-outBlue[close1, 1])**2+(outBlue[Cz, 2]-outBlue[close1, 2])**2)
        distNz[1]=np.sqrt((outBlue[Cz, 0]-outBlue[close2, 0])**2+(outBlue[Cz, 1]-outBlue[close2, 1])**2+(outBlue[Cz, 2]-outBlue[close2, 2])**2)
        if np.argmin(distNz)==0:
            Nz=close1
        else:
            Nz=close2
        Iz=opposite
    elif twoPoints=="Iz":
        distIz=np.zeros(2)
        distIz[0]=np.sqrt((outBlue[Cz, 0]-outBlue[close1, 0])**2+(outBlue[Cz, 1]-outBlue[close1, 1])**2+(outBlue[Cz, 2]-outBlue[close1, 2])**2)
        distIz[1]=np.sqrt((outBlue[Cz, 0]-outBlue[close2, 0])**2+(outBlue[Cz, 1]-outBlue[close2, 1])**2+(outBlue[Cz, 2]-outBlue[close2, 2])**2)
        if np.argmin(distIz)==0:
            Iz=close1
        else:
            Iz=close2
        Nz=opposite

    #Determining Rpa and Lra from cross product of known points
    idx2=np.zeros(2)
    step=0
    for i in range(numElBlue):
        #use only undetermined points
        if i!=close1 and i!= close2 and i!=opposite and i!=Cz:
            idx2[step]=i
            step+=1
    cr=np.cross(outBlue[Nz, :]-outBlue[Cz, :], outBlue[Iz, :]-outBlue[Cz, :])
    cr=cr/np.linalg.norm(cr)
    #if the result of the dot product is positive, the first point is Lpa, otherwise it is Rpa
    if np.dot(cr, outBlue[int(idx2[0]), :])>0:
        Lpa=int(idx2[0])
        Rpa=int(idx2[1])
    else:
        Lpa=int(idx2[1])
        Rpa=int(idx2[0])


    orderedBlue=np.zeros((5, 3))
    orderedBlue[0, :]=outBlue[Nz, :]
    orderedBlue[1, :]=outBlue[Iz, :]
    orderedBlue[2, :]=outBlue[Rpa, :]
    orderedBlue[3, :]=outBlue[Lpa, :]
    orderedBlue[4, :]=outBlue[Cz, :]

    return  orderedBlue, np.array([Nz, Iz, Rpa, Lpa, Cz])

def orderReferencePoints(numElYellow, numElBlue, outBlue, outYellow):
    '''
    #for this function to work, 5 blue stickers are needed;
    #it works only if the mean distance to the 5 closest yellow points from every blue point is smallest for Iz
    #Cz is the point with lowest std of distances to other blue points
    #Iz is the point with closest mean distance to 5 closest optode locations
    #Nz is the point with largest distance to Iz
    #Determining Rpa and Lra from cross product of known points

    input:
        numElYellow: number of yellow stickers
        numElBlue: number of blue points=5
        outBlue: points from blue stickers (5, 3)
        outYellow: points from yellow stickers (numElYellow, 3)
    output:
        orderedBlue: ordered points from blue stickers (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
    '''
    #Cz is the point with lowest std of distances to other blue points
    distRef=np.zeros((numElBlue, numElBlue))
    stds=np.zeros(numElBlue)
    arr=np.zeros(4)
    for i in range(numElBlue):
        step=0
        for j in range(numElBlue):
            distRef[i, j]=np.sqrt((outBlue[i, 0]-outBlue[j, 0])**2+(outBlue[i, 1]-outBlue[j, 1])**2+(outBlue[i, 2]-outBlue[j, 2])**2)
            if i!=j:
                arr[step]=distRef[i, j]
                step+=1

        stds[i]=np.std(arr[:])
    Cz=np.argmin(stds)

    #Iz is the point with closest mean distance to 5 closest optode locations
    idx0=np.zeros(4)
    step=0
    distOther=np.zeros((4, numElYellow))
    means=np.zeros(4)
    for i in range(numElBlue):
        #use only undetermined points
        if i!=Cz:
            idx0[step]=i
            #calculate distance to all yellow points
            for j in range(numElYellow):
                distOther[step, j]=np.sqrt((outBlue[i, 0]-outYellow[j, 0])**2+(outBlue[i, 1]-outYellow[j, 1])**2+(outBlue[i, 2]-outYellow[j, 2])**2)
            distOther[step, :]=np.sort(distOther[step, :])
            #calculate mean distance to 5 closest yellow points
            means[step]=np.mean(distOther[step, :5])
            step+=1
    Iz=int(idx0[np.argmin(means)])

    #Nz is the point with largest distance to Iz
    idx1=np.zeros(3)
    step=0
    distRef2=np.zeros(3)
    for i in range(numElBlue):
        #use only undetermined points
        if i!=Iz and i!=Cz:
            idx1[step]=i
            #distance to Iz
            distRef2[step]=np.sqrt((outBlue[i, 0]-outBlue[Iz, 0])**2+(outBlue[i, 1]-outBlue[Iz, 1])**2+(outBlue[i, 2]-outBlue[Iz, 2])**2)
            step+=1
    Nz=int(idx1[np.argmax(distRef2)])

    #Determining Rpa and Lra from cross product of known points
    idx2=np.zeros(2)
    step=0
    for i in range(numElBlue):
        #use only undetermined points
        if i!=Nz and i!=Iz and i!=Cz:
            idx2[step]=i
            step+=1
    cr=np.cross(outBlue[Nz, :]-outBlue[Cz, :], outBlue[Iz, :]-outBlue[Cz, :])
    cr=cr/np.linalg.norm(cr)
    #if the result of the dot product is positive, the first point is Lpa, otherwise it is Rpa
    if np.dot(cr, outBlue[int(idx2[0]), :])>0:
        Lpa=int(idx2[0])
        Rpa=int(idx2[1])
    else:
        Lpa=int(idx2[1])
        Rpa=int(idx2[0])


    orderedBlue=np.zeros((5, 3))
    orderedBlue[0, :]=outBlue[Nz, :]
    orderedBlue[1, :]=outBlue[Iz, :]
    orderedBlue[2, :]=outBlue[Rpa, :]
    orderedBlue[3, :]=outBlue[Lpa, :]
    orderedBlue[4, :]=outBlue[Cz, :]

    return  orderedBlue, np.array([Nz, Iz, Rpa, Lpa, Cz])
    

def alignPoints(numElYellow,  orderedBlue, outYellow, refPoints, use):
    '''
    input:
        numElYellow: number of yellow stickers
        orderedBlue: ordered points from blue stickers (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        outYellow: points from yellow stickers (numElYellow, 3)
        refPoints: reference points for alignment (5, 3)
        use: points corresponding to out (numElYellow, 3)
    output:
        trOrderedYellow: transformed ordered points from yellow stickers: to fit reference points (numElYellow, 3)->order is determined by points in use input
        trOrderedBlue: transformedordered points from blue stickers: to fit reference points (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        errLabels: error of each point to closest point in use input(1, numElYellow)->order is determined by points in use input   
    '''
    #align points to reference points
    myR, myT = rigidTransform3D(orderedBlue.T, refPoints.T)
    #transform blue and yellow points
    trOrderedBlue=myR@orderedBlue.T+myT
    trOrderedBlue=trOrderedBlue.T
    trOrderedYellow=myR@outYellow.T+myT
    trOrderedYellow=trOrderedYellow.T

    err=np.zeros((numElYellow, numElYellow))
    err2=np.zeros((numElYellow, 2))
    #calculate distance to all yellow points from use input
    for i in range(numElYellow):
        for j in range(numElYellow):
            err[i, j]=np.sqrt((trOrderedYellow[i, 0]-use[j, 0])**2+(trOrderedYellow[i, 1]-use[j, 1])**2+(trOrderedYellow[i, 2]-use[j, 2])**2)
        #find the closest point and index of the point it is closest to
        err2[i, 0]=min(err[i, :])
        err2[i, 1]=np.argmin(err[i, :])

    #check if each point is closest to a different point
    if len(err2[:, 1]) != len(set(err2[:, 1])):
        print('Error: some points are closest to the same point')
    #same order of points as in labels2->labels from colin model used in the montage
    orderedTrOut=np.zeros_like(trOrderedYellow)
    errLabels=np.zeros(numElYellow)
    for i in range(numElYellow):
        orderedTrOut[i, :]=trOrderedYellow[np.where(err2[:, 1]==i), :]
        errLabels[i]=err2[np.where(err2[:, 1]==i), 0]

    return trOrderedYellow, trOrderedBlue, errLabels

def writePointsTxt(numSor, numDet,  orderedBlue, outYellow, readFile, writeFile):
    '''
    input:
        numSor: number of sources (int)
        numDet: number of detectors (int)
        orderedBlue: ordered points from blue stickers (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        outYellow: points from yellow stickers (numElYellow, 3)
        readFile: name of the file to read the coordinates to align to = Standard_Optodes.txt file from the used montage; reference points need to be labeled O01-O05 in order Nz, Iz, Rpa, Lpa, Cz (string)
        filename: name of the file to write to (string)
    output:
        orderedTrOut: transformed ordered points from yellow stickers: to fit reference points (numElYellow, 3)->order is determined by points in use input
        orderedTrOutBlue: transformedordered points from blue stickers: to fit reference points (5, 3)->order: Nz, Iz, Rpa, Lpa, Cz
        errLabels: error of each point to closest point in use input(1, numElYellow)->order is determined by points in use input

    example: 
        orderedYellow, orderedBlue, errLabels=writePointsTxt(31, 30, orderedBlue, outYellow, "Standard_Optodes.txt", "writeFile.txt")
    '''
    #read .txt file
    content=np.loadtxt(readFile, dtype='str', delimiter=',')
    #reference label in input file
    labelsO=["O01", "O02", "O03", "O04", "O05"]
    #define reference and use points
    refPoints=np.zeros((5, 3))
    for i in range(5):
        refPoints[i, :]=np.array(content[np.where(content[:, 0]==labelsO[i]), 1:], dtype='float')
    numElYellow=numSor+numDet
    use=np.zeros((numSor+numDet, 3))
    for i in range(numSor+numDet):
        use[i, :]=np.array(content[i, 1:], dtype='float')
    #define labels
    labelsRef=["Nz", "Iz", "Rpa", "Lpa", "Cz"]

    #get affine transformation from reference points to blue points
    trn = affineFit(refPoints, orderedBlue)
    #use transformations to transform use points
    trUse=trn.Transform(use.T)
    trUse=np.array(trUse).T
    #use icp for better alignment so the corresponding points are found
    trUse2, _ = icp(outYellow, trUse)
    #calculate error to find the correspoinding points between locations and transposed use points
    err=np.zeros((numElYellow, numElYellow))
    err2=np.zeros((numElYellow, 2))
    for i in range(numElYellow):
        for j in range(numElYellow):
            err[i, j]=np.sqrt((outYellow[i, 0]-trUse2[j, 0])**2+(outYellow[i, 1]-trUse2[j, 1])**2+(outYellow[i, 2]-trUse2[j, 2])**2)
        err2[i, 0]=min(err[i, :])
        err2[i, 1]=np.argmin(err[i, :])

    #check if each point is closest to a different point
    if len(err2[:, 1]) != len(set(err2[:, 1])):
        print('Error: some points are closest to the same point')
        print(len(err2[:, 1]) , len(set(err2[:, 1])))
    orderedTrOut=np.zeros_like(outYellow)
    #same order of points as in labels2->labels from colin model used in the montage
    errLabels=np.zeros(numElYellow)
    for i in range(numElYellow):
        orderedTrOut[i, :]=outYellow[np.where(err2[:, 1]==i), :]
        errLabels[i]=err2[np.where(err2[:, 1]==i), 0]
    #write in txt file
    with open(writeFile, 'w') as f:
        for i in range(5):
            f.write(labelsRef[i]+',{}, {}, {}'.format(orderedBlue[i, 0], orderedBlue[i, 1], orderedBlue[i, 2]))
            f.write('\n')
        for i in range(numElYellow):
            f.write(content[i, 0]+',{}, {}, {}'.format(orderedTrOut[i, 0], orderedTrOut[i, 1], orderedTrOut[i, 2]))
            f.write('\n')
    return orderedTrOut, orderedBlue, errLabels

def processOne(readFileScan, readFileOptodes, readFileClustersBlue, readFileClustersYellow, readFilePlaneBlue, readFilePlaneYellow, writeFile, masks, numSor, numDet, numElYellow, numElBlue, 
               radius, radiusBlue, myPath, optSizeYellow, optSizeBlue, distance2Plane, twoPoints, printErr=True, plot=False, idxArr=[], incorrectRecognition=False, splitFirst=False):
    '''
    input:
        readFileScan: scan file (string)
        readFileOptodes: optodes file (string)
        readFileClustersBlue: clusters file for blue stickers (string)
        readFileClustersYellow: clusters file for yellow stickers (string)
        readFilePlaneBlue: plane file for blue stickers (string)
        readFilePlaneYellow: plane file for yellow stickers (string)
        writeFile: name of the file to write to (string)
        masks: masks for yellow and blue stickers (list of 12 floats between 0 and 1)
        numSor: number of sources (int)
        numDet: number of detectors (int)
        numElYellow: number of yellow stickers (int)
        numElBlue: number of blue stickers (int)
        radius: size of the circle to fit to the yellow stickers (float)
        radiusBlue: size of the circle fit to the blue stickers (float)
        myPath: path to the folder containing the files (WindowsPath)
        optSizeYellow: size to subtract from the yellow stickers to get wanted point (float)
        optSizeBlue: size to subtract from the blue stickers to get wanted point (float)
        distance2Plane: distance from fitted plane to keep points in clusters (float)
        two points: label where there are two stickers together (string "Nz" or "Iz")
        printErr: print mean and std of distance from aligned points to points in Standard_Optodes.txt (bool)
        plot: plot the alignment (bool)
        idxArr: if the ordere is set wrong by the algorithm, the user can input the order; order of the blue stickers (list of 5 ints)
        incorrectRecognition: if the order is set wrong by the algorithm, the user can input the order; which scan is incorrect (bool)
        splitFirst: if the 2 blue stickers are put too close, split first cluster (list)
    output:
        orderedYellow: ordered yellow stickers (numElYellow, 3)
        orderedBlue: ordered blue stickers (numElBlue, 3)
        errLabels: error with same order as orderedYellow (1, numElYellow)

    example:
        orderedYellow, orderedBlue, errLabels=f.processAll("scan/scan2post"", "Standard_Optodes.txt", "scan/normals(+d)Blue{num}.csv".format(num={}), "scan/normals(+d)Yellow{num}.csv".format(num={}), 
        "scan/normals(+d)Blue{num}.csv".format(num={}), "scan{scan}/normals(+d)Yellow{num}.csv".format(num={}), "coordsTest.txt", [0.14 0.035, 0.65, 0.35, 0.8, 0.3, 0.35, 0.1, 0.45, 0.35, 0.45, 0.15], 31, 30, 61, 6, 
        6.5, 5, pathlib.Path().resolve(), 22.6, 0, 2, "Iz", printErr=True, plot=False, idxArr=[], incorrectRecognition=False, splitFirst=False)
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
    MaskYellow=(np.abs(vcolors_hsv[:,0] - yHueCenter) < yHueWidth) & (np.abs(vcolors_hsv[:,1] - ySatCenter) < ySatWidth) & (np.abs(vcolors_hsv[:,2] - yValueCenter) < yValueWidth)
    MaskBlue=(np.abs(vcolors_hsv[:,0] - bHueCenter) < bHueWidth) & (np.abs(vcolors_hsv[:,1] - bSatCenter) < bSatWidth) & (np.abs(vcolors_hsv[:,2] - bValueCenter) < bValueWidth)
    #get points in mask
    newPointsYellow=pointsiInMask(vnew, MaskYellow)
    newPointsBlue=pointsiInMask(vnew, MaskBlue)
    #if the recognition of the ordered points is incorrect, user should input the order of the reference points; this happens if another cluster had more detected points in the mask than reference stickers
    if incorrectRecognition:
        numElBlue=np.max(idxArr)+1
        orderedBlue=np.zeros((5, 3))
    #get as many clusters as there are stickers
    finalClustersYellow=finalClusters(newPointsYellow, eps=radius, minSamples=50, numEl=numElYellow)
    finalClustersBlue=finalClusters(newPointsBlue, eps=radiusBlue, minSamples=50, numEl=numElBlue)
    #if the 6th blue sticker was too close to another one and the clusters were merged, split the first cluster
    if splitFirst:            
        finalClustersBlue0=finalClusters(finalClustersBlue[0], eps=3, minSamples=50, numEl=2)
        finalClustersBlue[5]=finalClustersBlue[4]
        finalClustersBlue[4]=finalClustersBlue[3]
        finalClustersBlue[3]=finalClustersBlue[2]
        finalClustersBlue[2]=finalClustersBlue[1]
        finalClustersBlue[1]=finalClustersBlue0[1]
        finalClustersBlue[0]=finalClustersBlue0[0]
    #write points belonging to each cluster to a file
    writePoints(finalClustersYellow, readFileClustersYellow)
    writePoints(finalClustersBlue, readFileClustersBlue)
    #fit a plane to each cluster
    getPlane(finalClustersYellow, myPath, numElYellow, readFileClustersYellow, readFilePlaneYellow, diameter=2*radius)
    getPlane(finalClustersBlue, myPath, numElBlue, readFileClustersBlue, readFilePlaneBlue, diameter=2*radius)
    #cut points too far from the plane and rewrite the points file
    writePointsNearPlane(numElYellow, readFileClustersYellow, readFilePlaneYellow, myPath, distance2Plane)
    writePointsNearPlane(numElBlue, readFileClustersBlue, readFilePlaneBlue, myPath, distance2Plane)
    #re-fit the plane
    getPlane(finalClustersYellow, myPath, numElYellow, readFileClustersYellow, readFilePlaneYellow, diameter=2*radius)
    getPlane(finalClustersBlue, myPath, numElBlue, readFileClustersBlue, readFilePlaneBlue,  diameter=2*radius)
    #subtract optode size from each point to get a point on the surface of the head
    outYellow=subtractOptode(readFilePlaneYellow, numElYellow, optSizeYellow)
    outBlue=subtractOptode(readFilePlaneBlue, numElBlue, optSizeBlue)
    #if the order is pre-defined, use it
    if incorrectRecognition:
        for i in range(5):
            orderedBlue[i, :]=outBlue[idxArr[i], :]
    else:
        #order the blue points, use a different function for 6 blue stickers
        #5 blue points function works for montages, where Iz has the lowest mean distance to closest 5 yellow points
        if numElBlue==5:
            orderedBlue, order=orderReferencePoints(numElYellow, numElBlue, outBlue, outYellow)
        elif numElBlue==6:
            orderedBlue, order=orderReferencePoints6(numElBlue, outBlue, twoPoints)
    #returns almost the same results
    #outBlue=subtractCapThickness(outBlue, numElBlue, scan, order, 0.5, ["Cz", "Iz"]) 
    #orderedBlue, _=orderReferencePoints(numElYellow, numElBlue, outBlue, outYellow)
    #wrtie the final file and return ordered yellow points, ordered blue points and error to closest point in input use
    orderedYellow, orderedBlue, errLabels=writePointsTxt(numSor, numDet, orderedBlue, outYellow, readFileOptodes, writeFile)
    #print mean and std of error
    if printErr==True:
        print(np.mean(errLabels[:]), np.std(errLabels[:]))
    #plot the last alignment
    if plot==True:
        #read .txt file to get reference and use points
        content=np.loadtxt(readFileOptodes, dtype='str', delimiter=',')
        #reference label in input file
        labelsO=["O01", "O02", "O03", "O04", "O05"]
        #define reference and use points
        refPoints=np.zeros((5, 3))
        for i in range(5):
            refPoints[i, :]=np.array(content[np.where(content[:, 0]==labelsO[i]), 1:], dtype='float')
        use=np.zeros((numSor+numDet, 3))
        for i in range(numSor+numDet):
            use[i, :]=np.array(content[i, 1:], dtype='float')
        #get affine transfroamtion
        trn = affineFit(refPoints, orderedBlue)
        #use transformations to transform reference points and use points
        trRefPoints=trn.Transform(refPoints.T)
        trRefPoints=np.array(trRefPoints).T
        trUse=trn.Transform(use.T)
        trUse=np.array(trUse).T
        #use icp for better alignment so the corresponding points are found
        trUse, _ = icp(outYellow, trUse)
        #plot alignment
        fig = p.figure(figsize=(15, 15)); 
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(5):
            ax.scatter(refPoints[i, 0], refPoints[i, 1], refPoints[i, 2], s=10, color='b')
            ax.scatter(orderedBlue[i, 0], orderedBlue[i, 1], orderedBlue[i, 2], s=10, color='g')

        ax=fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(5):
            ax.scatter(trRefPoints[i, 0], trRefPoints[i, 1], trRefPoints[i, 2], s=10, color='b')
            ax.scatter(orderedBlue[i, 0], orderedBlue[i, 1], orderedBlue[i, 2], s=10, color='r')

        fig = p.figure(figsize=(15, 15)); 
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(numElYellow):
            ax.scatter(use[i, 0], use[i, 1], use[i, 2], s=10, color='b')
            ax.scatter(outYellow[i, 0], outYellow[i, 1], outYellow[i, 2], s=10, color='g')

        ax=fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(numElYellow):
            ax.scatter(trUse[i, 0], trUse[i, 1], trUse[i, 2], s=10, color='b')
            ax.scatter(outYellow[i, 0], outYellow[i, 1], outYellow[i, 2], s=10, color='r')

    return orderedYellow, orderedBlue, errLabels

def processAll(subject, masks, scans, bestScan, numElYellow, numElBlue, radius, radiusBlue, myPath, optSizeYellow, optSizeBlue, distance2Plane, twoPoints="Nz", 
               printErr=True, plot=False, idxArr=[], incorrectRecognition=[], splitFirst=[]):
    """
    used for processing a serier of scans for one subject - aligning the scans to one of them
    running script needs to be in a folder containing folders with subjects' names; each subject folder contains folders with scans numbered 0 to scans-1;
    each scan folder contains a .obj and a .jpg file with the same name;

    input:
        subject: subject name (string)
        masks: masks for yellow and blue stickers (list of 12 floats between 0 and 1)
        scans: number of scans (int)
        bestScan: best scan used for alignment of others, blue stickers are well seen (int)
        numElYellow: number of yellow stickers (int)
        numElBlue: number of blue stickers (int)
        radius: size of the circle to fit to the yellow stickers (float)
        radiusBlue: size of the circle fit to the blue stickers (list of #numElBlue floats)
        myPath: path to the folder containing the files (WindowsPath)
        optSizeYellow: size to subtract from the yellow stickers to get wanted point (float)
        optSizeBlue: size to subtract from the blue stickers to get wanted point (float)
        distance2Plane: distance from fitted plane to keep points in clusters (float)
        two points: label where there are two stickers together (string "Nz" or "Iz")
        printErr: print mean and std of error (bool)
        plot: plot the last alignment (bool)
        idxArr: if the ordere is set wrong by the algorithm, the user can input the order; order of the blue stickers (list of 5 ints)
        incorrectRecognition: if the order is set wrong by the algorithm, the user can input the order; which scan is incorrect (list of ints)
        splitFirst: if the 2 blue stickers are put too close, split first cluster (list)
    output:
        orderedYellow: transformed ordered yellow stickers to fit reference points = best scan blue points (numElYellow, 3, scans)
        orderedBlue: transformed ordered blue stickers to fit reference points = best scan blue points (numElBlue, 3, scans)
        errLabels: error with same order as orderedYellow (numElYellow, scans)

    example: 
        trOrderedYellow, trOrderedBlue, errLabels=processAll("Filip", [0.14 0.035, 0.65, 0.35, 0.8, 0.3, 0.35, 0.1, 0.45, 0.35, 0.45, 0.15], 8, 2, 61, 6, 
        6.5, [5, 5, 5, 5, 5, 5, 5, 5], pathlib.Path().resolve(), 22.6, 0, 2, "Iz"
    """

    #color mask
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

    trOrderedYellow=np.zeros((numElYellow, 3, scans))
    trOrderedBlue=np.zeros((5, 3, scans))
    errLabels=np.zeros((numElYellow, scans))
    step=0
    for scan in range(scans):
        #first iteration goes through the best scan indicated by the input and is used as refence for other scans
        if scan==0:
            scan=bestScan
        elif scan<=bestScan:
            scan=scan-1
        else:
            scan=scan
        #reset number of blue points
        numElBlueAlt=numElBlue
        #file to write and read clusters and planes; the second {} gets filled in in the function
        readFilePlaneYellow=subject+"/scan{scan}/normals(+d)Yellow{num}.csv".format(scan=scan, num={})
        readFilePlaneBlue=subject+"/scan{scan}/normals(+d)Blue{num}.csv".format(scan=scan, num={})
        readFileClustersBlue=subject+"/scan{scan}/pointsBlue{num}.ply".format(scan=scan, num={})
        readFileClustersYellow=subject+"/scan{scan}/pointsYellow{num}.ply".format(scan=scan, num={})
        #get vertex locations and colors
        vnew, vcolors_rgb, vcolors_hsv=preProcessing(subject+"/scan{}/scan{}post".format(scan, scan))
        #define mask
        MaskYellow=(np.abs(vcolors_hsv[:,0] - yHueCenter) < yHueWidth) & (np.abs(vcolors_hsv[:,1] - ySatCenter) < ySatWidth) & (np.abs(vcolors_hsv[:,2] - yValueCenter) < yValueWidth)
        MaskBlue=(np.abs(vcolors_hsv[:,0] - bHueCenter) < bHueWidth) & (np.abs(vcolors_hsv[:,1] - bSatCenter) < bSatWidth) & (np.abs(vcolors_hsv[:,2] - bValueCenter) < bValueWidth)
        #get points in mask
        newPointsYellow=pointsiInMask(vnew, MaskYellow)
        newPointsBlue=pointsiInMask(vnew, MaskBlue)
        #if the recognition of the ordered points is incorrect, user should input which scan is wrong and the order of the blue points; this happened a couple times where another cluster had more detected points in the mask than reference points
        if scan in incorrectRecognition:
            numElBlueAlt=np.max(idxArr[step])+1
            orderedBlue=np.zeros((5, 3))
        #get as many clusters as there are stickers
        finalClustersYellow=finalClusters(newPointsYellow, eps=radius, minSamples=50, numEl=numElYellow)
        finalClustersBlue=finalClusters(newPointsBlue, eps=radiusBlue[scan], minSamples=50, numEl=numElBlueAlt)
        #in 2 sets of scans, the 6th blue sticker was too close to another one and the clusters were merged
        #6th sticker is used to determine one of the points, followed by the rest
        if scan in splitFirst:            
            finalClustersBlue0=finalClusters(finalClustersBlue[0], eps=3, minSamples=50, numEl=2)
            finalClustersBlue[5]=finalClustersBlue[4]
            finalClustersBlue[4]=finalClustersBlue[3]
            finalClustersBlue[3]=finalClustersBlue[2]
            finalClustersBlue[2]=finalClustersBlue[1]
            finalClustersBlue[1]=finalClustersBlue0[1]
            finalClustersBlue[0]=finalClustersBlue0[0]
        #write points belonging to each cluster to a file
        writePoints(finalClustersYellow, readFileClustersYellow)
        writePoints(finalClustersBlue, readFileClustersBlue)
        #fit a plane to each cluster
        getPlane(finalClustersYellow, myPath, numElYellow, readFileClustersYellow, readFilePlaneYellow, diameter=2*radius)
        getPlane(finalClustersBlue, myPath, numElBlueAlt, readFileClustersBlue, readFilePlaneBlue, diameter=2*radius)
        #cut points too far from the plane and rewrite the points file
        writePointsNearPlane(numElYellow, readFileClustersYellow, readFilePlaneYellow, myPath, distance2Plane)
        writePointsNearPlane(numElBlueAlt, readFileClustersBlue, readFilePlaneBlue, myPath, distance2Plane)
        #re-fit the plane
        getPlane(finalClustersYellow, myPath, numElYellow, readFileClustersYellow, readFilePlaneYellow, diameter=2*radius)
        getPlane(finalClustersBlue, myPath, numElBlueAlt, readFileClustersBlue, readFilePlaneBlue, diameter=2*radius)
        #subtract optode size from each point to get a point on the surface of the head
        outYellow=subtractOptode(readFilePlaneYellow, numElYellow, optSizeYellow)
        outBlue=subtractOptode(readFilePlaneBlue, numElBlueAlt, optSizeBlue)
        #if the order is pre-defined, use it
        if scan in incorrectRecognition:
            for i in range(5):
                orderedBlue[i, :]=outBlue[idxArr[step][i], :]
            step+=1
        else:
            #order the blue points, use a different function for 6 blue points
            #5 blue points function works for montages, where Iz has the lowest mean distance to closest 5 yellow points
            if numElBlue==5:
                orderedBlue, order=orderReferencePoints(numElYellow, numElBlue, outBlue, outYellow)
            elif numElBlue==6:
                orderedBlue, order=orderReferencePoints6(numElBlue, outBlue, twoPoints)
        #in the first iteration, use points as reference points for the other scans
        if scan==bestScan:
            refPoints=orderedBlue
            use=outYellow
        #returns almost the same results
        #outBlue=subtractCapThickness(outBlue, numElBlue, scan, order, 0.5, ["Cz", "Iz"]) 
        #orderedBlue, _=orderReferencePoints(numElYellow, numElBlue, outBlue, outYellow)
        #align the points to the reference points, return ordered points and error to closest point
        trOrderedYellow[:, :, scan], trOrderedBlue[:, :, scan], errLabels[:, scan]=alignPoints(numElYellow, orderedBlue, outYellow, refPoints, use)
        #print mean and std of error
        if printErr==True:
            print(np.mean(errLabels[:, scan]), np.std(errLabels[:, scan]))
    #plot the last alignment
    if plot==True:
        fig = p.figure(figsize=(15, 15)); 
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(numElBlue):
            ax.scatter(refPoints[i, 0], refPoints[i, 1], refPoints[i, 2], s=10, color='b')
            ax.scatter(orderedBlue[i, 0], orderedBlue[i, 1], orderedBlue[i, 2], s=10, color='g')

        ax=fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(numElBlue):
            ax.scatter(refPoints[i, 0], refPoints[i, 1], refPoints[i, 2], s=10, color='b')
            ax.scatter(trOrderedBlue[i, 0], trOrderedBlue[i, 1], trOrderedBlue[i, 2], s=10, color='r')

        fig = p.figure(figsize=(15, 15)); 
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        for i in range(numElYellow):
            ax.scatter(use[i, 0], use[i, 1], use[i, 2], s=10, color='b')
            ax.scatter(outYellow[i, 0], outYellow[i, 1], outYellow[i, 2], s=10, color='g')

        ax=fig.add_subplot(1, 2, 2, projection="3d")
        for i in range(numElYellow):
            ax.scatter(use[i, 0], use[i, 1], use[i, 2], s=10, color='b')
            ax.scatter(trOrderedYellow[i, 0], trOrderedYellow[i, 1], trOrderedYellow[i, 2], s=10, color='r')

    return trOrderedYellow, trOrderedBlue, errLabels

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

