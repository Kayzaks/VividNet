
import numpy as np
import math
import time
from enum import Enum
from numba import cuda
from GraphicsUserInterface import GraphicsUserInterface
from RendererKernel import mainRender
from Capsule import Capsule
from AttributePool import AttributePool
from Attribute import Attribute
from AttributeType import AttributeLexical


class Shapes(Enum):
    Box         = 0
    Sphere      = 1
    
    def __int__(self):
        return self.value
    

class ShapesRenderer:

    def __init__(self):
        self._x = 0


    def setShapeAttributePool(self, attributePool : AttributePool):
        # Prepositions
        attributePool.createType("X-Pos", AttributeLexical.Preposition)
        attributePool.createType("Y-Pos", AttributeLexical.Preposition)
        attributePool.createType("Z-Pos", AttributeLexical.Preposition)
        
        attributePool.createType("X-Rot", AttributeLexical.Preposition)
        attributePool.createType("Y-Rot", AttributeLexical.Preposition)
        attributePool.createType("Z-Rot", AttributeLexical.Preposition)
        
        attributePool.createType("X-Size", AttributeLexical.Preposition)
        attributePool.createType("Y-Size", AttributeLexical.Preposition)
        attributePool.createType("Z-Size", AttributeLexical.Preposition)

        # Textures        
        # TODO: Correct Textures
        attributePool.createType("R-Color", AttributeLexical.Adjective)
        attributePool.createType("G-Color", AttributeLexical.Adjective)
        attributePool.createType("B-Color", AttributeLexical.Adjective)

        # Lighting
        attributePool.createType("Light-R-Color", AttributeLexical.NonTransmit)
        attributePool.createType("Light-G-Color", AttributeLexical.NonTransmit)
        attributePool.createType("Light-B-Color", AttributeLexical.NonTransmit)
        
        attributePool.createType("Light-X-Dir", AttributeLexical.NonTransmit)
        attributePool.createType("Light-Y-Dir", AttributeLexical.NonTransmit)
        attributePool.createType("Light-Z-Dir", AttributeLexical.NonTransmit)


        # Background
        for uu in range(3):
            for vv in range(3):
                attributePool.createType("DCTR-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)
                attributePool.createType("DCTG-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)
                attributePool.createType("DCTB-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)


    def createAttributesForShape(self, shape : Shapes, capsule : Capsule, attributePool : AttributePool):
        # TODO: According to shape
        capsule.createAttribute("X-Pos", attributePool)
        capsule.createAttribute("Y-Pos", attributePool)
        capsule.createAttribute("Z-Pos", attributePool)
        
        capsule.createAttribute("X-Rot", attributePool)
        capsule.createAttribute("Y-Rot", attributePool)
        capsule.createAttribute("Z-Rot", attributePool)
        
        capsule.createAttribute("X-Size", attributePool)
        capsule.createAttribute("Y-Size", attributePool)
        capsule.createAttribute("Z-Size", attributePool)

        # Textures      
        # TODO: Correct Textures  
        capsule.createAttribute("R-Color", attributePool)
        capsule.createAttribute("G-Color", attributePool)
        capsule.createAttribute("B-Color", attributePool)

        # Lighting
        capsule.createAttribute("Light-R-Color", attributePool)
        capsule.createAttribute("Light-G-Color", attributePool)
        capsule.createAttribute("Light-B-Color", attributePool)

        capsule.createAttribute("Light-X-Dir", attributePool) 
        capsule.createAttribute("Light-Y-Dir", attributePool)
        capsule.createAttribute("Light-Z-Dir", attributePool)

        # Background
        for uu in range(3):
            for vv in range(3):
                capsule.createAttribute("DCTR-" + str(uu) + "-" + str(vv), attributePool)
                capsule.createAttribute("DCTG-" + str(uu) + "-" + str(vv), attributePool)
                capsule.createAttribute("DCTB-" + str(uu) + "-" + str(vv), attributePool)



    def renderShape(self, shape : Shapes, capsule : Capsule, width : int, height : int):
        # Attributes:
        # 0     -> Shapes
        # 1-3   -> Rotation Sin             = sin(*-Rot)
        # 4-6   -> Rotation Cos             = cos(*-Rot)
        # 7-9   -> Size                     = *-Size
        # 10-12 -> Position                 = *-Pos
        # 13-21 -> DCTR[0,0] - DCTR[2,2]    = DCTR-*-*
        # 22-30 -> DCTG[0,0] - DCTG[2,2]    = DCTG-*-*
        # 31-39 -> DCTB[0,0] - DCTB[2,2]    = DCTB-*-*
        # 40-42 -> Color                    = *-Color

        transferAttributes = np.empty(49, dtype=np.float32)

        transferAttributes[0] = float(int(shape))

        # These strange axis assignments are due to differing coordinate systems
        transferAttributes[1] = math.sin(math.pi * 0.5 * capsule.getAttributeValue("Z-Rot"))
        transferAttributes[4] = math.cos(math.pi * 0.5 * capsule.getAttributeValue("Z-Rot"))
        transferAttributes[2] = math.sin(math.pi * 0.5 * capsule.getAttributeValue("Y-Rot"))
        transferAttributes[5] = math.cos(math.pi * 0.5 * capsule.getAttributeValue("Y-Rot"))
        transferAttributes[3] = math.sin(math.pi * 0.5 * (capsule.getAttributeValue("X-Rot") - 0.5))
        transferAttributes[6] = math.cos(math.pi * 0.5 * (capsule.getAttributeValue("X-Rot") - 0.5))

        transferAttributes[7] = capsule.getAttributeValue("X-Size")
        transferAttributes[8] = capsule.getAttributeValue("Y-Size")
        transferAttributes[9] = capsule.getAttributeValue("Z-Size")
        
        transferAttributes[10] = capsule.getAttributeValue("X-Pos")
        transferAttributes[11] = capsule.getAttributeValue("Y-Pos")
        transferAttributes[12] = capsule.getAttributeValue("Z-Pos")

        index = 0
        for vv in range(3):
            for uu in range(3):

                transferAttributes[13 + index] = capsule.getAttributeValue("DCTR-" + str(uu) + "-" + str(vv))
                transferAttributes[22 + index] = capsule.getAttributeValue("DCTG-" + str(uu) + "-" + str(vv))
                transferAttributes[31 + index] = capsule.getAttributeValue("DCTB-" + str(uu) + "-" + str(vv))

                index = index + 1
            
            
        transferAttributes[40] = capsule.getAttributeValue("Light-R-Color")
        transferAttributes[41] = capsule.getAttributeValue("Light-G-Color")
        transferAttributes[42] = capsule.getAttributeValue("Light-B-Color")
        
        transferAttributes[43] = capsule.getAttributeValue("Light-X-Dir")
        transferAttributes[44] = capsule.getAttributeValue("Light-Y-Dir")
        transferAttributes[45] = capsule.getAttributeValue("Light-Z-Dir")

        transferAttributes[46] = capsule.getAttributeValue("R-Color")
        transferAttributes[47] = capsule.getAttributeValue("G-Color")
        transferAttributes[48] = capsule.getAttributeValue("B-Color")
      
        deviceAttributes = cuda.to_device(transferAttributes)

        data = np.zeros((width * height, 4), dtype=np.float32)
        threadsperblock = 32 
        blockspergrid = (width * height + (threadsperblock - 1)) // threadsperblock

        for xa in range(1):
            start = time.time()
            mainRender[blockspergrid, threadsperblock](data, width, height, deviceAttributes)
            end = time.time()
            print("Time needed for Render: " + str(end - start))

        pixels = np.zeros((height, width, 3), dtype=float)
        depths = np.zeros((height, width, 1), dtype=float)
        for xx in range(width):
            for yy in range(height):
                pixels[height - 1 - yy][xx][0] = data[xx * height + yy][0]
                pixels[height - 1 - yy][xx][1] = data[xx * height + yy][1]
                pixels[height - 1 - yy][xx][2] = data[xx * height + yy][2]
                depths[height - 1 - yy][xx] = data[xx * height + yy][0]

        newGUI = GraphicsUserInterface()

        newGUI.drawArray(pixels, width, height)