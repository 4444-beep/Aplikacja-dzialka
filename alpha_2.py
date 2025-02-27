
import imutils 
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from shapely import to_geojson
import numpy as np
from shapely import Polygon 
import os
import time
start_time = time.time()
def sort_coordinates(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(-angles)
    return list_of_xy_coords[indices]
 



def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def dim_drone_alt(image):
    #angles=[41.5,90,48.5] for dji mini 2
    b=input("podaj wysokość podczas robienia zdjęcia")
    #from tables
    sin_415=0.61256015297 
    cos_415=0.79042397419
    #calculates width of area in photo
    c=b/cos_415
    a=sin_415*c
    a2=a*a
    
    A2=image.shape[1]
    return a2, A2

#option 1
#width=dji_min_2()
#option2
cenX=[]
cenY=[]
width=0.05
img_name="0"
th=3
upper_orange= np.array([0, 215, 255], dtype = "uint8")

lower_orange= np.array([0, 140, 255], dtype = "uint8")




while os.path.exists(img_name)==False or img_name=="0":
    img_name=input("--- podaj ścieżkę zdjęcia ---\n")
    if os.path.exists(img_name)==True:
        image = cv2.imread(img_name)
        img_copy=cv2.imread(img_name)
        


mask = cv2.inRange(image, lower_orange, upper_orange)

image=cv2.bitwise_and(image, image, mask = mask)
	

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (1, 1), 0)





# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=10)
edged = cv2.erode(edged, None, iterations=10)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and, then initialize the
# distance colors and reference object
colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),(255, 0, 255))



refObj = None
#tu 

for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 1:
		continue
	# compute the rotated bounding box of the contour
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order, then draw the outline of the rotated bounding box
	box = perspective.order_points(box)
	# compute the center of the bounding box
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])
	
	# if this is the first contour we are examining (i.e., the left-most contour), we presume this is the reference object if this is the first 
    # contour we are examining (i.e.,the left-most contour), we presume this is the reference object
	if refObj is None:
		# unpack the ordered bounding box, then compute the midpoint between the top-left and top-right points, followed by the midpoint between the top-right and bottom-right
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		# compute the Euclidean distance between the midpoints,
		# then construct the reference object
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		refObj = (box, (cX, cY), D / width)
		
		cenX.append(int(cX))
		cenY.append(int(cY))
		
		continue
	# draw the contours on the image

	cv2.drawContours(img_copy, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(img_copy, [refObj[0].astype("int")], -1, (0, 255, 0), 2)


   
	
	cenX.append(int(cX))
	cenY.append(int(cY))



x=zip(cenX,cenY)
 
pgon = Polygon(x)



cords=np.array(pgon.exterior.coords)
cords=sort_coordinates(cords)


for i in range(0,len(cords)-1,1):
    pnts1=cords[i]
    x1=int(pnts1[0])
    y1=int(pnts1[1])
    pnts2=cords[i+1]
    x2=int(pnts2[0])
    y2=int(pnts2[1])
    cv2.line(img_copy,(x1,y1),(x2,y2),colors[2],th)
    D = dist.euclidean((x1,y1), (x2, y2)) / refObj[2]
    (mX, mY) = midpoint((x1,y1), (x2, y2))
    cv2.putText(img_copy, "{:.1f}m".format(D), (int(mX), int(mY)),
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[3], 2)

pnts1=cords[len(cords)-1]
x1=int(pnts1[0])
y1=int(pnts1[1])
pnts2=cords[0]
x2=int(pnts2[0])
y2=int(pnts2[1])
cv2.line(img_copy,(x1,y1),(x2,y2),colors[2],th)

D = dist.euclidean((x1,y1), (x2, y2)) / refObj[2]

(mX, mY) = midpoint((x1,y1), (x2, y2))
cv2.putText(img_copy, "{:.1f}m".format(D), (int(mX), int(mY - 10)),
cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[3], 2)
    




pow=str((pgon.area)/D*D)
obw=str((pgon.length)/D)

cv2.imshow("Image", img_copy)
#print(shapely.to_geojson(pgon))
f=open("dane.geojson","a")
f.write(to_geojson(pgon))
f.close()

print("--- powierzchnia to :"+pow+" m2 ---")
print("--- obwód to :"+obw+" m ---")
print("--- zapisano do pliku dane.geojson ---")
print("--- czas działania %s sekund ---" % (time.time() - start_time))
cv2.waitKey(0)
