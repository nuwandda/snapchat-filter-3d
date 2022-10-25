import cv2
import numpy as np
import math


def constrainPoint(p, w, h):
  """Constrains points to be inside boundary

  Args:
      p (tuple): Point to be constrained
      w (int): Width
      h (int): Height

  Returns:
      tuple: Returns constrained point
  """
  p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
  
  return p


def similarityTransform(inPoints, outPoints):
  """Compute similarity transform given two sets of two points. OpenCV requires 3 pairs of corresponding points. We are faking the third one.

  Args:
      inPoints (list): Point to transform
      outPoints (list): Point to transform

  Returns:
      list: Returns estimated point
  """
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()

  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

  inPts.append([np.int(xin), np.int(yin)])

  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

  outPts.append([np.int(xout), np.int(yout)])

  # Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
  return tform[0]


def rectContains(rect, point):
  """Check if a point is inside a rectangle

  Args:
      rect (list): Rectangle coordinates
      point (list): Point coordinates

  Returns:
      bool: Returns True if the given point lies in rectangle
  """
  if point[0] < rect[0]:
    return False
  elif point[1] < rect[1]:
    return False
  elif point[0] > rect[2]:
    return False
  elif point[1] > rect[3]:
    return False
  return True
