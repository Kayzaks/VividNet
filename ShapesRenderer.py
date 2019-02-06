
import numpy as np
import math
import time
import random
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

    def setPixelLayerAttributePool(self, attributePool : AttributePool, width : int, height : int):
        for xx in range(width):
            for yy in range(height):
                attributePool.createType("PixelR-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelG-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelB-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)

    def createAttributesForPixelLayer(self, width : int, height : int, capsule : Capsule, attributePool : AttributePool):
        self.setPixelLayerAttributePool(attributePool, width, height)
        
        for xx in range(width):
            for yy in range(height):
                capsule.createAttribute("PixelR-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelG-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelB-" + str(xx) + "-" + str(yy), attributePool)
    
    def getLambdaGOutputMap(self, shape : Shapes, capsule : Capsule, width : int, height : int):
        mapIdxAttr : dict = {}  # Index - Attribute
        mapAttrIdx : dict = {}  # Attribute - Index

        index = 0
        for yy in range(height):
            for xx in range(width):
                mapIdxAttr[index]     = capsule.getAttributeByName("PixelR-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 1] = capsule.getAttributeByName("PixelG-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 2] = capsule.getAttributeByName("PixelB-" + str(xx) + "-" + str(yy))
                index = index + 3

        for key, value in mapIdxAttr.items():
            mapAttrIdx[value] = key

        return (mapIdxAttr, mapAttrIdx)

    def setShapeAttributePool(self, attributePool : AttributePool):
        # Prepositions        
        attributePool.createType("X-Rot", AttributeLexical.Preposition)
        attributePool.createType("Y-Rot", AttributeLexical.Preposition)
        attributePool.createType("Z-Rot", AttributeLexical.Preposition)
        
        attributePool.createType("X-Size", AttributeLexical.Preposition)
        attributePool.createType("Y-Size", AttributeLexical.Preposition)
        attributePool.createType("Z-Size", AttributeLexical.Preposition)

        attributePool.createType("X-Pos", AttributeLexical.Preposition)
        attributePool.createType("Y-Pos", AttributeLexical.Preposition)
        attributePool.createType("Z-Pos", AttributeLexical.Preposition)

        # Background
        for uu in range(3):
            for vv in range(3):
                attributePool.createType("DCTR-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)
                attributePool.createType("DCTG-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)
                attributePool.createType("DCTB-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)

        # Lighting
        attributePool.createType("Light-R-Color", AttributeLexical.NonTransmit)
        attributePool.createType("Light-G-Color", AttributeLexical.NonTransmit)
        attributePool.createType("Light-B-Color", AttributeLexical.NonTransmit)
        
        attributePool.createType("Light-X-Dir", AttributeLexical.NonTransmit)
        attributePool.createType("Light-Y-Dir", AttributeLexical.NonTransmit)
        attributePool.createType("Light-Z-Dir", AttributeLexical.NonTransmit)

        # Textures        
        # TODO: Correct Textures
        attributePool.createType("R-Color", AttributeLexical.Adjective)
        attributePool.createType("G-Color", AttributeLexical.Adjective)
        attributePool.createType("B-Color", AttributeLexical.Adjective)
        
        # Shading
        # TODO: Transmit as "Metallic", etc
        attributePool.createType("Diffuse-Power",     AttributeLexical.NonTransmit)
        attributePool.createType("Diffuse-R-Color",   AttributeLexical.NonTransmit)
        attributePool.createType("Diffuse-G-Color",   AttributeLexical.NonTransmit)
        attributePool.createType("Diffuse-B-Color",   AttributeLexical.NonTransmit)
        attributePool.createType("Specular-Exponent", AttributeLexical.NonTransmit)
        attributePool.createType("Specular-Power",    AttributeLexical.NonTransmit)
        attributePool.createType("Specular-R-Color",  AttributeLexical.NonTransmit)
        attributePool.createType("Specular-G-Color",  AttributeLexical.NonTransmit)
        attributePool.createType("Specular-B-Color",  AttributeLexical.NonTransmit)
        attributePool.createType("Fresnel-Exponent",  AttributeLexical.NonTransmit)
        attributePool.createType("Ambient-Power",     AttributeLexical.NonTransmit)
        attributePool.createType("Ambient-R-Color",   AttributeLexical.NonTransmit)
        attributePool.createType("Ambient-G-Color",   AttributeLexical.NonTransmit)
        attributePool.createType("Ambient-B-Color",   AttributeLexical.NonTransmit)


    def createAttributesForShape(self, shape : Shapes, capsule : Capsule, attributePool : AttributePool):
        self.setShapeAttributePool(attributePool)
        
        # TODO: According to shape        
        capsule.createAttribute("X-Rot", attributePool)
        capsule.createAttribute("Y-Rot", attributePool)
        capsule.createAttribute("Z-Rot", attributePool)
        
        capsule.createAttribute("X-Size", attributePool)
        capsule.createAttribute("Y-Size", attributePool)
        capsule.createAttribute("Z-Size", attributePool)

        capsule.createAttribute("X-Pos", attributePool)
        capsule.createAttribute("Y-Pos", attributePool)
        capsule.createAttribute("Z-Pos", attributePool)

        # Background
        for uu in range(3):
            for vv in range(3):
                capsule.createAttribute("DCTR-" + str(uu) + "-" + str(vv), attributePool)
                capsule.createAttribute("DCTG-" + str(uu) + "-" + str(vv), attributePool)
                capsule.createAttribute("DCTB-" + str(uu) + "-" + str(vv), attributePool)

        # Lighting
        capsule.createAttribute("Light-R-Color", attributePool)
        capsule.createAttribute("Light-G-Color", attributePool)
        capsule.createAttribute("Light-B-Color", attributePool)

        capsule.createAttribute("Light-X-Dir", attributePool) 
        capsule.createAttribute("Light-Y-Dir", attributePool)
        capsule.createAttribute("Light-Z-Dir", attributePool)

        # Textures      
        # TODO: Correct Textures  
        capsule.createAttribute("R-Color", attributePool)
        capsule.createAttribute("G-Color", attributePool)
        capsule.createAttribute("B-Color", attributePool)

        # Shading
        capsule.createAttribute("Diffuse-Power",     attributePool)
        capsule.createAttribute("Diffuse-R-Color",   attributePool)
        capsule.createAttribute("Diffuse-G-Color",   attributePool)
        capsule.createAttribute("Diffuse-B-Color",   attributePool)
        capsule.createAttribute("Specular-Exponent", attributePool)
        capsule.createAttribute("Specular-Power",    attributePool)
        capsule.createAttribute("Specular-R-Color",  attributePool)
        capsule.createAttribute("Specular-G-Color",  attributePool)
        capsule.createAttribute("Specular-B-Color",  attributePool)
        capsule.createAttribute("Fresnel-Exponent",  attributePool)
        capsule.createAttribute("Ambient-Power",     attributePool)
        capsule.createAttribute("Ambient-R-Color",   attributePool)
        capsule.createAttribute("Ambient-G-Color",   attributePool)
        capsule.createAttribute("Ambient-B-Color",   attributePool)


    def getLambdaGInputMap(self, shape : Shapes, capsule : Capsule):
        mapIdxAttr : dict = {}  # Index - Attribute
        mapAttrIdx : dict = {}  # Attribute - Index

        mapIdxAttr[0] = capsule.getAttributeByName("X-Rot")
        mapIdxAttr[1] = capsule.getAttributeByName("Y-Rot")
        mapIdxAttr[2] = capsule.getAttributeByName("Z-Rot")
        
        mapIdxAttr[3] = capsule.getAttributeByName("X-Size")
        mapIdxAttr[4] = capsule.getAttributeByName("Y-Size")
        mapIdxAttr[5] = capsule.getAttributeByName("Z-Size")

        mapIdxAttr[6] = capsule.getAttributeByName("X-Pos")
        mapIdxAttr[7] = capsule.getAttributeByName("Y-Pos")
        mapIdxAttr[8] = capsule.getAttributeByName("Z-Pos")

        # Background
        index = 0
        for uu in range(3):
            for vv in range(3):
                mapIdxAttr[9  + index] = capsule.getAttributeByName("DCTR-" + str(uu) + "-" + str(vv))
                mapIdxAttr[18 + index] = capsule.getAttributeByName("DCTG-" + str(uu) + "-" + str(vv))
                mapIdxAttr[27 + index] = capsule.getAttributeByName("DCTB-" + str(uu) + "-" + str(vv))
                index = index + 1

        # Lighting
        mapIdxAttr[36] = capsule.getAttributeByName("Light-R-Color")
        mapIdxAttr[37] = capsule.getAttributeByName("Light-G-Color")
        mapIdxAttr[38] = capsule.getAttributeByName("Light-B-Color")

        mapIdxAttr[39] = capsule.getAttributeByName("Light-X-Dir") 
        mapIdxAttr[40] = capsule.getAttributeByName("Light-Y-Dir")
        mapIdxAttr[41] = capsule.getAttributeByName("Light-Z-Dir")

        # Textures      
        # TODO: Correct Textures  
        mapIdxAttr[42] = capsule.getAttributeByName("R-Color")
        mapIdxAttr[43] = capsule.getAttributeByName("G-Color")
        mapIdxAttr[44] = capsule.getAttributeByName("B-Color")

        # Shading
        mapIdxAttr[45] = capsule.getAttributeByName("Diffuse-Power")
        mapIdxAttr[46] = capsule.getAttributeByName("Diffuse-R-Color")
        mapIdxAttr[47] = capsule.getAttributeByName("Diffuse-G-Color")
        mapIdxAttr[48] = capsule.getAttributeByName("Diffuse-B-Color")
        mapIdxAttr[49] = capsule.getAttributeByName("Specular-Exponent")
        mapIdxAttr[50] = capsule.getAttributeByName("Specular-Power")
        mapIdxAttr[51] = capsule.getAttributeByName("Specular-R-Color")
        mapIdxAttr[52] = capsule.getAttributeByName("Specular-G-Color")
        mapIdxAttr[53] = capsule.getAttributeByName("Specular-B-Color")
        mapIdxAttr[54] = capsule.getAttributeByName("Fresnel-Exponent")
        mapIdxAttr[55] = capsule.getAttributeByName("Ambient-Power")
        mapIdxAttr[56] = capsule.getAttributeByName("Ambient-R-Color")
        mapIdxAttr[57] = capsule.getAttributeByName("Ambient-G-Color")
        mapIdxAttr[58] = capsule.getAttributeByName("Ambient-B-Color")

        for key, value in mapIdxAttr.items():
            mapAttrIdx[value] = key

        return (mapIdxAttr, mapAttrIdx)

    def renderInputGenerator(self, shape : Shapes, width : int, height : int):
        outList = np.random.rand(59)

        # Fixate Randoms or correlate those that should

        # We ignore Position for now
        outList[6] = 0.0
        outList[7] = 0.0
        outList[8] = 0.0

        # Current shapes have uniform size larger than 0.2
        outList[3] = max(outList[3], 0.2)
        outList[4] = outList[3]
        outList[5] = outList[3]

        # Test on white light        
        outList[36] = 1.0
        outList[37] = 1.0
        outList[38] = 1.0

        return outList


    def renderShape(self, shape : Shapes, attributes : list, width : int, height : int, showDebug : bool = False):
        # Input Attributes as Lambda G Mapping
        # 
        # Transfer Attributes:
        # 0     -> Shapes
        # 1-3   -> Rotation Sin             = sin(*-Rot)
        # 4-6   -> Rotation Cos             = cos(*-Rot)
        # 7-9   -> Size                     = *-Size
        # 10-12 -> Position                 = *-Pos
        # 13-21 -> DCTR[0,0] - DCTR[2,2]    = DCTR-*-*
        # 22-30 -> DCTG[0,0] - DCTG[2,2]    = DCTG-*-*
        # 31-39 -> DCTB[0,0] - DCTB[2,2]    = DCTB-*-*
        # 40-42 -> Light Color              = Light-*-Color
        # 43-45 -> Light Direction          = Light-*-Dir
        # 46-48 -> Color                    = *-Color
        # 49    -> Diffuse Power            = Diffuse-Power
        # 50-52 -> Diffuse Color            = Diffuse-*-Color
        # 53    -> Specular Exponent        = Specular-Exponent
        # 54    -> Specular Power           = Specular-Power
        # 55-57 -> Specular Color           = Specular-*-Color
        # 58    -> Fresnel Exponent         = Fresnel-Exponent
        # 59    -> Ambient Power            = Ambient-Power
        # 60-62 -> Ambient Color            = Ambient-*-Color

        transferAttributes = np.empty(63, dtype=np.float32)

        transferAttributes[0] = float(int(shape))

        for index in range(len(attributes)):
            transferAttributes[index + 4] = attributes[index]

        # These strange axis assignments are due to differing coordinate systems
        transferAttributes[1] = math.sin(math.pi * 0.5 * attributes[2])
        transferAttributes[4] = math.cos(math.pi * 0.5 * attributes[2])
        transferAttributes[2] = math.sin(math.pi * 0.5 * attributes[1])
        transferAttributes[5] = math.cos(math.pi * 0.5 * attributes[1])
        transferAttributes[3] = math.sin(math.pi * 0.5 * (attributes[0] - 0.5))
        transferAttributes[6] = math.cos(math.pi * 0.5 * (attributes[0] - 0.5))
        
        # allow DCT's to be negative
        for index in range(13, 40):
            transferAttributes[index] = ( transferAttributes[index] - 0.5 ) * 4.0

        # 360° Lighting
        length = np.linalg.norm([transferAttributes[43] - 0.5, transferAttributes[44] - 0.5, transferAttributes[45] - 0.5])
        transferAttributes[43] = (transferAttributes[43] - 0.5) / length
        transferAttributes[44] = (transferAttributes[44] - 0.5) / length
        transferAttributes[45] = (transferAttributes[45] - 0.5) / length

        # Shading        
        transferAttributes[49] = transferAttributes[49] * 2.0
        transferAttributes[53] = transferAttributes[53] * transferAttributes[53] * 100.0
        transferAttributes[54] = transferAttributes[54] * 10.0
        transferAttributes[58] = transferAttributes[58] * 1.3


      
        deviceAttributes = cuda.to_device(transferAttributes)

        data = np.zeros((width * height, 4), dtype=np.float32)
        threadsperblock = 32 
        blockspergrid = (width * height + (threadsperblock - 1)) // threadsperblock

        start = time.time()
        mainRender[blockspergrid, threadsperblock](data, width, height, deviceAttributes)
        end = time.time()

        if showDebug is True:
            print("Time needed for Render: " + str(end - start))

        pixels = np.zeros(height * width * 3, dtype=float)

        indx = 0
        for xx in range(width):
            for yy in range(height):
                pixels[(yy * width + xx) * 3]     = min(max(data[xx * height + yy][0], 0.0), 1.0)
                pixels[(yy * width + xx) * 3 + 1] = min(max(data[xx * height + yy][1], 0.0), 1.0)
                pixels[(yy * width + xx) * 3 + 2] = min(max(data[xx * height + yy][2], 0.0), 1.0)

        return pixels

        '''pixels = np.zeros((height, width, 3), dtype=float)
        depths = np.zeros((height, width, 1), dtype=float)

        for xx in range(width):
            for yy in range(height):
                pixels[height - 1 - yy][xx][0] = data[xx * height + yy][0]
                pixels[height - 1 - yy][xx][1] = data[xx * height + yy][1]
                pixels[height - 1 - yy][xx][2] = data[xx * height + yy][2]
                depths[height - 1 - yy][xx] = data[xx * height + yy][0]

        return (pixels, depths)'''