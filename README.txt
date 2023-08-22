Filip Jenko, august 2023

the script processes a scan of a subject's head while wearing cap with fNIRS optodes; 
the color stickers on top of the optodes (as well as at the reference points - Nz, Iz, Rpa, Lpa, Cz) are located and the coordinates of the points, where the optodes touch the subject's head are written in the output. 
input files are the .obj and .jpg files from the scan, the file Standard_Optodes.txt (contains coordinates of the points from the montage, which are used for alignment) and the inputs written in testScript
the script outputs a .txt file with the labels (sources, detectors or reference points) and their coordinates

the .obj file is too big for github, so download it from this link: 
https://drive.google.com/drive/folders/134eX9eocCyplw8JEKZgCUbtrGzL6bgeG?usp=sharing
it needs to be placed in the scan folder

make sure you have all the modules needed by the functions installed

scan folder - .obj and .jpg file of the scan
actuallySelectingColors - selecting color masks to use in the script; %matplotlib widget is used for spinning a 3D scatter plot; the line can be deleted
coordsTest.txt - test output
functionsFinal - functions used
Standard_Optodes.txt - coordinates of the montage to align to
testScript - running script; %matplotlib widget is used for spinning a 3D scatter plot; the line can be deleted

montage needed:
when you have a montage you want, you should add the reference points to the montage; this can be done in NIRSite: first load your montage, add reference points as optodes labeled as other: Nz, Iz, Rpa, Lpa and Cz should be labeled as  O01, O02, O03, O04, O05 in order
after this, save the montage; the script needs the file Standard_Optodes.txt, so this file should be in the same folder as the scirpt; if it is not, you can change the path in the input readFileOptodes

before scanning:
6 stickers (green in the example) for reference;
5 of them are placed on Nz, Iz, Rpa, Lpa, Cz; 6th one is placed below Iz (distance from 6th sticker to Cz needs to be bigger than distance from Cz to Iz); the sticker should be roughly on the same plane as Iz, Nz, and Cz; it will be used for determening, which point is which
in this example yellow stickers are used for optode detection; if you use another color, change the mask boundaries accordingly; there is another notebook (actuallySelectingColors) for visualisation and setting the boundaries
user should be careful and make sure, cables are not on top of the any sticker, so the scanner can detect them well 
the script works even if the stickers are not that well seen, if there is only half of the sticker visible, results are accordingly worse

after scanning:
what is needed: an .obj and a .jpg file from a scan; the scan needs to be processed (ExStar - scanner software) already and the user should confirm that color stickers can be well seen in the mesh; 
Exstar software can be used for generating a mesh from the pointcloud; it also generates the .obj and .jpg files, which are used in the script
 
