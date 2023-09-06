Filip Jenko, august 2023

General
=======

  * the script processes a scan of a subject's head while wearing cap with fNIRS optodes;
  * the color stickers on top of the optodes (as well as at the reference points - Nz, Iz, Rpa, Lpa, Cz) are located with this script
  * the coordinates and labels of the points, where the optodes touch the subject's head are written in the output (alongside the mentioned reference points)
  * points detected as the sticekrs on top of optodes are regarded to as optode points; points detected as the sticekrs on top of Nz, Iz, Rpa, Lpa, Cz are reference points
  * input files are the .obj and .jpg files from the scan, the file Standard_Optodes.txt (contains coordinates of the points from the montage, which are used for alignment) and the inputs written in testScript
  * the script outputs a .txt file with the labels (sources, detectors or reference points) and their coordinates

Running the script
==================

  * pull the git repository
  * create an environment using the file environment.yml
( if you are using anaconda, go to anaconda prompt, navigate to the pulled directory, and run the command: conda env create -f environment.yml )
  * download the .jpg and .obj files from the link: (don't forget to unzip)
https://drive.google.com/drive/folders/134eX9eocCyplw8JEKZgCUbtrGzL6bgeG (google drive)
  * it needs to be placed in the scan folder
  * run the testScript

Code procedure
==============

  * in preprocessing, the .obj and .jpg files are converted to matrices of scanned point coordinates and their hsv values (preProcessing)
  * scanned points are filtered with the input color masks to get points from the yellow and green stickers (pointsInMask)
  * from the filtered points, big enough clusters are detected (as many as we have sticekers) (finalClusters)
  * points, belonging to sticker clusters are written in seperate files (for possible visualization in blender) (writePoints)
  * plane is fitted to every cluster (getPlane)
  * points, too far away from the plane are discarded (writePointsNearPlane)
  * new points, belonging to clusters are rewritten in the files (writePointsNearPlane)
  * plane is refitted to every cluster (getPlane)
  * size of the optode is subtracted from the fitted plane in the direction of the normal towards the head (subtractOptode)
  * determining which reference point is which (orderReferencePoints6)
(Iz and 6th sticker are closest together; Nz is furthest away from both of them; Cz lies roughly on the same plane as Nz, Iz and 6th sticker; Rpa and Lpa are determinted from cross product of th known points)
  * now we know which scanned reference point is which and which montage reference point is which and we can fit montage points to the scanned points  (writePointsTxt)
  * labels of the scanned points are determined by the label of the closest montage point; (writePointsTxt) 
  * labels and coordinates of the scanned points are written in the output (writePointsTxt)

In the flowchart below, rectangles represent functions and their names, text on top of arrows represents inputs and outputs from the functions.
![flowchart of the code procedure](https://github.com/FilipJenko/headmodelling/blob/main/images/flowchart1.png)

Visualising which coordinates are in the output
-----------------------------------------------

 1. open NIRSite and select adult head model
 2. click import and select: 
  * import from: register digitized positions
  * field separator: comma
  * units: mm
  * registration method: align (rotate, translate)
  * uncheck scan to scalp
 3. load the output file, which this script generated

If you notice that a point/sticker is wrongly detected (setting color mask)
---------------------------------------------------------------------------

  * check which cluster has enough points, so it is recognised as an optode sticker
  * to do that, open actuallySelectingColors, run first 2 cells to get data
  * run cell #YELLOW to see yellow points in the selected mask (#GREEN for green points)
  * visualize, where is the unwanted cluster of points is (you can plot all the points in the output to the same plot in the cell YELLOW)
  * modify the mask accordingly, (if the points in the extra cluster are too dark, they have either too low 2nd or 3rd coordinate (saturation or value) in the HSV color space - change that boundary and remove the cluster)
  * check if the rest of the points are detected correctly after the modification; if not try changing the other coordinate (saturation or value)
  * if you cannot get the masks to detect only the correct clusters, change the color of the used stickers
  * important when selecting a new color of the sticker: do a test scan with all of the colors on all sides of the head (same color stickers on different sides of the head, might have different lighting and consequently different HSV values)
  * choose the color, where you can separate the points of the stickers from the other points the easiest (use 3d scatter plot to detect a color cluster which is not connected to the rest of the points)

If you want to run the script for a different scan
--------------------------------------------------

  * load the new scan files in the scan folder (the .jpg and .obj files need to have the same name)
  * for running actuallySelectingColors script, change the name of the variable readFileScan to the name of the file without the extension (example: scan/name)
  * for running testScript, change the name of the variable readFileScan to the name of the file without the extension (example: scan/name)

If the new scan also has a different montage
--------------------------------------------

  * first create the new montage and save it
  * take the Standard_Optodes.txt file and copy it to the directory, where this script is running from; if you have several Standard_Optodes files, change the input readFileInputAlignment name if necessary
  * change the inputs numSor, numDet, numScannedOptodes if necessary

Content of the repository
=========================

  * scan folder - .obj and .jpg file of the scan
  * actuallySelectingColors - selecting color masks to use in the script; %matplotlib widget is used for spinning a 3D scatter plot; the line can be deleted
  * coordsTestColin2.txt - test output
  * environment.yml - a file to set up the needed environment with used packages
  * functionsFinal - functions used
  * Standard_Optodes.txt - coordinates of the montage reference points and montage optode points, which are aligned to the scanned points, in order to give the scanned points the correct label in the output
  * testScript - running script; %matplotlib widget is used for spinning a 3D scatter plot; the line can be deleted
  * unusedFunctions - functions, not used in this script but can be implemented if needed

Montage needed
==============

  * when you decide on a montage, you should add the reference points to the montage
  * standard montages can be downloaded from the NIRX support center

### Adding reference points can be done in NIRSite ###

  * first load your montage
  * add reference points as optodes labeled as other: Nz, Iz, Rpa, Lpa and Cz should be labeled as  O01, O02, O03, O04, O05 in order
ORDER OF THE SELECTED REFERENCE POINTS IS IMPORTANT FOR THE SCRIPT TO RUN CORRECTLY
  * after this, save the montage; the script needs the file Standard_Optodes.txt, so this file should be in the same folder as the script; if it is not, you can change the path in the input readFileOptodes

Scanning process
================

Before scanning
---------------

  * 6 stickers (green in the example) for reference
  * 5 of them are placed on Nz, Iz, Rpa, Lpa, Cz; 6th one is placed below Iz (distance from 6th sticker to Cz needs to be bigger than distance from Cz to Iz)
  * the sticker should be roughly on the same plane as Iz, Nz, and Cz; it will be used for determining, which point is which
  * in this example yellow stickers are used for optode detection; if you use another color, change the mask boundaries accordingly; there is another notebook (actuallySelectingColors) for visualization and setting the boundaries
  * user should be careful and make sure, cables are not on top of the any sticker, so the scanner can detect them well 
  * the script works even if the stickers are not that well seen, if there is only half of the sticker visible, results are accordingly worse

Setup
-----

  * try to create roughly equal lightning from all sides: close the window drapes and turn on the lights (I used two lights from the sides)
  * make sure to put optode cables out of the way so they don't cover the stickers
  * calibrate the scanner if necessary (see manual)
  * create a project in the ExStar software
  * I used portrait mode, hybrid alignment and the resolution of 0.5 mm
  * see scanner manual for how to scan

### Quick tips ###

  * instruct the subject to focus on a point in front of them and try to keep still
  * run on data quality indicator - you want the stickers to be colored green on the screen -  good detection
  * scan one side first and slowly move around so you see the detected point cloud expanding
  * don't go around the head too many times, since the subject will move the head and this will cause the data to be inaccurate
  * for me it worked best to make the scans about 3 mins long; focus on the parts of the head that you are interested in (stickers); 
  * scan one side first, then the top of the head, then the front of the head, other side, back side 
  * top of the head if important to scan, in the first half, in some cases when I didn' do that, it caused misalignment
  * don't move to an unmapped part of the head too quickly - this might cause misalignment and you would have to repeat the scan
  * be careful not to put fingers or scanner's wristband in front of the camera - misalignment
  * notice the distance indicator
  * whenever you notice misalignment (the currently scanned white points are not mapped correctly with regard to other points of the scan) - pause the scan and see if you can delete data, which is detected wrongly; otherwise restart the scan
  * I didn't have much success pausing and resuming the scans, since subject will usually move too much in the meantime

Post processing
---------------

  * see if the point cloud has sharp edges of the stickers - good detections
  * if there are double edges of the circle stickers, you can use lasso to select points to delete the unwanted points
  * I recommend duplicating the folder before deleting the points 
  * you can try to create a mesh, even if the result looks to have unnecessary data - sometimes it still turns out well
  * delete points from the neck down, so the processing is faster

### Creating a mesh ###

I used the settings: create watertight model
  * quality: high
  * filter: none
  * smooth: low
  * remove small floating parts: 0 (if you use this option, be sure not to delete floating stickers)
  * unchecked simplifications
  * unchecked max triangles
  * checked remove spikes

### After scanning ###

  * what is needed: an .obj and a .jpg file from a scan; the scan needs to be processed (ExStar - scanner software) already and the user should confirm that color stickers can be well seen in the mesh
  * ExStar software can be used for generating a mesh from the pointcloud; it also generates the .obj and .jpg files, which are used in the script
 
