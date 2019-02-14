
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
        self._DCTDimension = 28

    def setPixelLayerAttributePool(self, attributePool : AttributePool, width : int, height : int):
        for xx in range(width):
            for yy in range(height):
                attributePool.createType("PixelC-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelX-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)
                attributePool.createType("PixelY-" + str(xx) + "-" + str(yy), AttributeLexical.Pixel)

    def createAttributesForPixelLayer(self, width : int, height : int, capsule : Capsule, attributePool : AttributePool):
        self.setPixelLayerAttributePool(attributePool, width, height)
        
        for xx in range(width):
            for yy in range(height):
                capsule.createAttribute("PixelC-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelX-" + str(xx) + "-" + str(yy), attributePool)
                capsule.createAttribute("PixelY-" + str(xx) + "-" + str(yy), attributePool)
    
    def getLambdaGOutputMap(self, shape : Shapes, capsule : Capsule, width : int, height : int):
        mapIdxAttr : dict = {}  # Index - Attribute
        mapAttrIdx : dict = {}  # Attribute - Index

        index = 0
        for yy in range(height):
            for xx in range(width):
                mapIdxAttr[index]     = capsule.getAttributeByName("PixelC-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 1] = capsule.getAttributeByName("PixelX-" + str(xx) + "-" + str(yy))
                mapIdxAttr[index + 2] = capsule.getAttributeByName("PixelY-" + str(xx) + "-" + str(yy))
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

        # Lighting
        attributePool.createType("Light-Intensity", AttributeLexical.NonTransmit)
        
        attributePool.createType("Light-X-Dir", AttributeLexical.NonTransmit)
        attributePool.createType("Light-Y-Dir", AttributeLexical.NonTransmit)
        attributePool.createType("Light-Z-Dir", AttributeLexical.NonTransmit)

        # Textures        
        # TODO: Correct Textures
        attributePool.createType("Texture-Intensity", AttributeLexical.Adjective)
        
        # Shading
        # TODO: Transmit as "Metallic", etc
        attributePool.createType("Diffuse-Power",     AttributeLexical.NonTransmit)
        attributePool.createType("Specular-Exponent", AttributeLexical.NonTransmit)
        attributePool.createType("Specular-Power",    AttributeLexical.NonTransmit)
        attributePool.createType("View-Exponent",     AttributeLexical.NonTransmit)
        attributePool.createType("Ambient-Power",     AttributeLexical.NonTransmit)
        attributePool.createType("Reflection-Power",  AttributeLexical.NonTransmit)

        for uu in range(self._DCTDimension):
            for vv in range(self._DCTDimension):
                attributePool.createType("BackgroundDCT-" + str(uu) + "-" + str(vv), AttributeLexical.NonTransmit)
                attributePool.createType("ReflectDCT-" + str(uu) + "-" + str(vv),  AttributeLexical.NonTransmit)

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

        # Lighting
        capsule.createAttribute("Light-Intensity", attributePool)

        capsule.createAttribute("Light-X-Dir", attributePool) 
        capsule.createAttribute("Light-Y-Dir", attributePool)
        capsule.createAttribute("Light-Z-Dir", attributePool)

        # Textures      
        # TODO: Correct Textures  
        capsule.createAttribute("Texture-Intensity", attributePool)

        # Shading
        capsule.createAttribute("Diffuse-Power",     attributePool)
        capsule.createAttribute("Specular-Exponent", attributePool)
        capsule.createAttribute("Specular-Power",    attributePool)
        capsule.createAttribute("View-Exponent",  attributePool)
        capsule.createAttribute("Ambient-Power",     attributePool)
        capsule.createAttribute("Reflection-Power",  attributePool)
        
        # Background
        for uu in range(self._DCTDimension):
            for vv in range(self._DCTDimension):
                capsule.createAttribute("BackgroundDCT-" + str(uu) + "-" + str(vv), attributePool)
                capsule.createAttribute("ReflectDCT-" + str(uu) + "-" + str(vv),  attributePool)


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

        # Lighting
        mapIdxAttr[9] = capsule.getAttributeByName("Light-Intensity")

        mapIdxAttr[10] = capsule.getAttributeByName("Light-X-Dir") 
        mapIdxAttr[11] = capsule.getAttributeByName("Light-Y-Dir")
        mapIdxAttr[12] = capsule.getAttributeByName("Light-Z-Dir")

        # Textures      
        # TODO: Correct Textures  
        mapIdxAttr[13] = capsule.getAttributeByName("Texture-Intensity")

        # Shading
        mapIdxAttr[14] = capsule.getAttributeByName("Diffuse-Power")
        mapIdxAttr[15] = capsule.getAttributeByName("Specular-Exponent")
        mapIdxAttr[16] = capsule.getAttributeByName("Specular-Power")
        mapIdxAttr[17] = capsule.getAttributeByName("View-Exponent")
        mapIdxAttr[18] = capsule.getAttributeByName("Ambient-Power")
        mapIdxAttr[19] = capsule.getAttributeByName("Reflection-Power")
        

        # Background
        index = 0
        for uu in range(self._DCTDimension):
            for vv in range(self._DCTDimension):
                mapIdxAttr[20 + index] = capsule.getAttributeByName("ReflectDCT-" + str(uu) + "-" + str(vv))
                mapIdxAttr[20 + self._DCTDimension * self._DCTDimension + index] = capsule.getAttributeByName("BackgroundDCT-" + str(uu) + "-" + str(vv))
                index = index + 1


        for key, value in mapIdxAttr.items():
            mapAttrIdx[value] = key

        return (mapIdxAttr, mapAttrIdx)

    def renderInputGenerator(self, shape : Shapes, width : int, height : int):
        outList = np.random.rand(20 + self._DCTDimension * self._DCTDimension * 2)

        for i in range(len(outList)):
            outList[i] = max(outList[i], 0.1)

        # Fixate Randoms or correlate those that should

        # We ignore Position for now
        outList[6] = 0.0
        outList[7] = 0.0
        outList[8] = 0.0

        if (shape == Shapes.Sphere):
            # Ignore the rotation of a sphere
            outList[0] = 0.0
            outList[1] = 0.0
            outList[2] = 0.0


        # Current shapes have uniform size larger than 0.2
        outList[3] = min(max(outList[3], 0.5), 0.9)
        outList[4] = outList[3]
        outList[5] = outList[3]

        # Normalized Light
        length = np.linalg.norm([outList[10] - 0.5, outList[11] - 0.5, outList[12] - 0.5])
        outList[10] = (((outList[10] - 0.5) / length) / 2.0) + 0.5
        outList[11] = (((outList[11] - 0.5) / length) / 2.0) + 0.5
        outList[12] = (((outList[12] - 0.5) / length) / 2.0) + 0.5

        outList[13] = 0.0

        # Remove Reflections for now
        outList[19] = 0.0    

        for index in range(self._DCTDimension * self._DCTDimension):
                outList[20 + index] = 0.0

        return outList


    def getModelSplit(self, shape : Shapes):
        sqDCT = self._DCTDimension * self._DCTDimension
        #return [0, 20 + sqDCT * 2]
        return [0, 20, 20 + sqDCT * 2]


    def renderShape(self, shape : Shapes, attributes : list, width : int, height : int, altBackground = None, showDebug : bool = False):
        # Input Attributes as Lambda G Mapping
        # 
        # Transfer Attributes:
        # 0     -> Shapes
        # 1-3   -> Rotation Sin             = sin(*-Rot)
        # 4-6   -> Rotation Cos             = cos(*-Rot)
        # 7-9   -> Size                     = *-Size
        # 10-12 -> Position                 = *-Pos
        # 13    -> Light Intensity          = Light-Intensity
        # 14-16 -> Light Direction          = Light-*-Dir
        # 17    -> Texture Intensity        = Texture-Intensity
        # 18    -> Diffuse Power            = Diffuse-Power
        # 19    -> Specular Exponent        = Specular-Exponent
        # 20    -> Specular Power           = Specular-Power
        # 21    -> View Exponent            = View-Exponent
        # 22    -> Ambient Power            = Ambient-Power
        # 23    -> Reflection Power         = Reflection-Power
        # 24+   -> Reflect DCT[0,0]-[2,2]   = ReflectDCT-*-*
        #       -> DCT[0,0] - DCT[2,2]      = BackgroundDCT-*-*

        transferAttributes = np.empty(len(attributes) + 4, dtype=np.float32)

        transferAttributes[0] = float(int(shape))

        for index in range(len(attributes)):
            transferAttributes[index + 4] = attributes[index]

        # These strange axis assignments are due to differing coordinate systems
        transferAttributes[1] = math.sin(math.pi * 0.5 * attributes[2])
        transferAttributes[4] = math.cos(math.pi * 0.5 * attributes[2])
        transferAttributes[2] = math.sin(math.pi * 0.5 * attributes[1])
        transferAttributes[5] = math.cos(math.pi * 0.5 * attributes[1])
        transferAttributes[3] = math.sin(math.pi * 0.5 * attributes[0])
        transferAttributes[6] = math.cos(math.pi * 0.5 * attributes[0])
        
        # Shading        
        transferAttributes[18] = transferAttributes[18] * transferAttributes[18] * 5.0
        transferAttributes[19] = transferAttributes[19] * transferAttributes[19] * 100.0
        transferAttributes[20] = transferAttributes[20] * transferAttributes[20] * 1.0
        transferAttributes[21] = transferAttributes[21] * 2.0
        #transferAttributes[22] = transferAttributes[22] * 2.0
        transferAttributes[23] = transferAttributes[23] * 1.0

        
        # Redo Lighting       
        transferAttributes[14] = (transferAttributes[14] - 0.5) * 2.0
        transferAttributes[15] = (transferAttributes[15] - 0.5) * 2.0
        transferAttributes[16] = (transferAttributes[16] - 0.5) * 2.0
        


        # allow DCT's to be negative and in the range of -2048 to 2048 for the lowest frequency
        index = 0
        for uu in range(self._DCTDimension):
            for vv in range(self._DCTDimension):
                intensity = (4096.0 / ((max(uu, vv) + 1)))  # (1024.0 / math.pow(2, max(uu, vv)))

                # R,G,B,Reflection
                for colorIndex in range(2):
                    offset = 24 + colorIndex * self._DCTDimension * self._DCTDimension
                    transferAttributes[index + offset] = (transferAttributes[index + offset] - 0.5) * intensity
 
                index = index + 1


      
        deviceAttributes = cuda.to_device(transferAttributes)

        data = np.zeros((width * height, 4), dtype=np.float32)
        threadsperblock = 32 
        blockspergrid = (width * height + (threadsperblock - 1)) // threadsperblock

        start = time.time()
        mainRender[blockspergrid, threadsperblock](data, width, height, deviceAttributes, self._DCTDimension)
        end = time.time()

        if showDebug is True:
            print("Time needed for Render: " + str(end - start))

        pixels = np.zeros(height * width * 3, dtype=float)

        indx = 0
        for xx in range(width):
            for yy in range(height):

                if ( data[xx * height + yy][3] > 2) and altBackground is not None:
                    pixels[(yy * width + xx) * 3]     = altBackground[(yy * width + xx) * 3]
                else:
                    pixels[(yy * width + xx) * 3]     = min(max(data[xx * height + yy][0], 0.0), 1.0)

                pixels[(yy * width + xx) * 3 + 1] = float(xx) / float(width)
                pixels[(yy * width + xx) * 3 + 2] = float(yy) / float(height)

        return pixels

        '''pixels = np.zeros((height, width, 3), dtype=float)
        depths = np.zeros((height, width, 1), dtype=float)

        for xx in range(width):
            for yy in range(height):
                pixels[height - 1 - yy][xx][0] = data[xx * height + yy][0]
                pixels[height - 1 - yy][xx][1] = data[xx * height + yy][1]
                pixels[height - 1 - yy][xx][2] = data[xx * height + yy][2]
                depths[height - 1 - yy][xx] = data[xx * height + yy][3]

        return (pixels, depths)'''