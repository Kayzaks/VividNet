
import numpy as np
import math
import time
from numba import cuda, float64, float32, int32, int64
from GraphicsUserInterface import GraphicsUserInterface







@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def crossProduct(vecOut, vec1, vec2):
    vecOut[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    vecOut[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    vecOut[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]


@cuda.jit('float64(float64[:], float64[:])', device=True)
def dotProduct(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    
@cuda.jit('float64(float64[:])', device=True)
def vecNorm(vec1):
    return math.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])

@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def reflect(outVec, incident, normal):
    dx = dotProduct(incident, normal)
    outVec[0] = incident[0] - 2.0 * dx * normal[0]
    outVec[1] = incident[1] - 2.0 * dx * normal[1]
    outVec[2] = incident[2] - 2.0 * dx * normal[2]

@cuda.jit('void(float64[:], float64[:])', device=True)
def normalize(vecOut, vec1):
    dist = vecNorm(vec1)
    vecOut[0] = vec1[0] / dist
    vecOut[1] = vec1[1] / dist
    vecOut[2] = vec1[2] / dist
    
@cuda.jit('void(float64[:], float64[:, :], float64[:])', device=True)
def matMul(vecOut, matrix, vec):
    vecOut[0] = matrix[0, 0] * vec[0] + matrix[1, 0] * vec[1] + matrix[2, 0] * vec[2] 
    vecOut[1] = matrix[0, 1] * vec[0] + matrix[1, 1] * vec[1] + matrix[2, 1] * vec[2] 
    vecOut[2] = matrix[0, 2] * vec[0] + matrix[1, 2] * vec[1] + matrix[2, 2] * vec[2] 


@cuda.jit('void(float64[:, :], float64[:], float64[:], float64, float64[:])', device=True)
def setCamera(cameraToWorld, rayOrigin, ta, cr, auxVec):
    ta[0] = ta[0] - rayOrigin[0]
    ta[1] = ta[1] - rayOrigin[1]
    ta[2] = ta[2] - rayOrigin[2]
    normalize(ta, ta)
    auxVec[0] = math.sin(cr)
    auxVec[1] = math.cos(cr)
    auxVec[2] = 0.0

    crossProduct(cameraToWorld[0, :], ta, auxVec)
    normalize(cameraToWorld[0, :], cameraToWorld[0, :])
    
    crossProduct(cameraToWorld[1, :], cameraToWorld[0, :], ta)
    normalize(cameraToWorld[1, :], cameraToWorld[1, :])

    cameraToWorld[2,0] = ta[0]
    cameraToWorld[2,1] = ta[1]
    cameraToWorld[2,2] = ta[2]


@cuda.jit('void(float64[:], float64, float64)', device=True)
def clipVec(vecOut, leftBound, rightBound):
    if vecOut[0] > rightBound:
        vecOut[0] = rightBound
    elif vecOut[0] < leftBound:
        vecOut[0] = leftBound
    if vecOut[1] > rightBound:
        vecOut[1] = rightBound
    elif vecOut[1] < leftBound:
        vecOut[1] = leftBound
    if vecOut[2] > rightBound:
        vecOut[2] = rightBound
    elif vecOut[2] < leftBound:
        vecOut[2] = leftBound
        
@cuda.jit('float64(float64, float64, float64)', device=True)
def clip(value, leftBound, rightBound):
    if value > rightBound:
        return rightBound
    elif value < leftBound:
        return leftBound
    return value


@cuda.jit('float64(float64, float64, float64)', device=True)
def smoothstep(edge0, edge1, x):
    t = clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)



@cuda.jit('float64(float64[:])', device=True)
def sdPlane(point):
    return point[1]

@cuda.jit('float64(float64[:], float64)', device=True)
def sdSphere(point, radius):
    return vecNorm(point) - radius


@cuda.jit('float64(float64[:], float64, float64, float64)', device=True)
def sdBox(point, boundX, boundY, boundZ):
    point[0] = abs(point[0]) - boundX
    point[1] = abs(point[1]) - boundY
    point[2] = abs(point[2]) - boundZ
    outVal = min(max(point[0],max(point[1],point[2])),0.0)
    point[0] = max(point[0], 0.0)
    point[1] = max(point[1], 0.0)
    point[2] = max(point[2], 0.0)
    return outVal + vecNorm(point)



@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def opU(result, dist1, dist2):
    if dist1[0] < dist2[0]:
        result[0] = dist1[0]
        result[1] = dist1[1]
    else:
        result[0] = dist2[0]
        result[1] = dist2[1]


@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def worldMap(result, position, attributes):
    vec3d = cuda.local.array(shape=3, dtype=float64)

    vec3d[0] = position[0] - 0.0
    vec3d[1] = position[1] - 1.0
    vec3d[2] = position[2] - 0.0

    # Rotation X-Axis
    TransformTemp = vec3d[0] * attributes[4] - vec3d[1] * attributes[1]
    vec3d[1] = vec3d[0] * attributes[1] + vec3d[1] * attributes[4]
    vec3d[0] = TransformTemp

    # Rotation Y-Axis
    TransformTemp = vec3d[0] * attributes[5] + vec3d[2] * attributes[2]
    vec3d[2] = -vec3d[0] * attributes[2] + vec3d[2] * attributes[5]
    vec3d[0] = TransformTemp
    
    # Rotation Z-Axis
    TransformTemp = vec3d[1] * attributes[6] - vec3d[2] * attributes[3]
    vec3d[2] = vec3d[1] * attributes[3] + vec3d[2] * attributes[6]
    vec3d[1] = TransformTemp

    if attributes[0] > 0.5:
        result[0] = sdBox(vec3d, 0.3 * attributes[7], 0.3 * attributes[7], 0.3 * attributes[7])
    else:
        result[0] = sdSphere(vec3d, 0.5 * attributes[7])

    result[1] = 2.0

    


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:])', device=True)
def castRay(result, rayOrigin, rayDirection, attributes):
    tmin = 1.0
    tmax = 20.0
    
    pos = cuda.local.array(shape=3, dtype=float64)

    # bounding volume
    tp1 = (0.0-rayOrigin[1]) / rayDirection[1] 
    if tp1 > 0.0:
        tmax = min( tmax, tp1 )
    tp2 = (1.6-rayOrigin[1]) / rayDirection[1]
    if tp2 > 0.0: 
        if rayOrigin[1] > 1.6 :
            tmin = max( tmin, tp2 )
        else:
            tmax = min( tmax, tp2 )
    
    t = tmin
    m = -1.0
    for i in range(64):
        precis = 0.0004*t

        pos[0] = rayOrigin[0] + rayDirection[0] * t
        pos[1] = rayOrigin[1] + rayDirection[1] * t
        pos[2] = rayOrigin[2] + rayDirection[2] * t

        worldMap(result, pos, attributes)

        if result[0] < precis or t > tmax:
            break
        t += result[0]
        m = result[1]

    if t > tmax: 
        m=-1.0
    
    result[0] = t
    result[1] = m

# http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def calcNormal(outVec, position, attributes):
    auxVec = cuda.local.array(shape=3, dtype=float64)

    ex = 0.5773 * 0.0005

    auxVec[0] = position[0] + ex
    auxVec[1] = position[1] - ex
    auxVec[2] = position[2] - ex
    worldMap(outVec, auxVec, attributes)
    dx1 = outVec[0]

    auxVec[0] = position[0] - ex
    auxVec[1] = position[1] - ex
    auxVec[2] = position[2] + ex
    worldMap(outVec, auxVec, attributes)
    dx2 = outVec[0]

    auxVec[0] = position[0] - ex
    auxVec[1] = position[1] + ex
    auxVec[2] = position[2] - ex
    worldMap(outVec, auxVec, attributes)
    dx3 = outVec[0]

    auxVec[0] = position[0] + ex
    auxVec[1] = position[1] + ex
    auxVec[2] = position[2] + ex
    worldMap(outVec, auxVec, attributes)
    dx4 = outVec[0]

    outVec[0] =   ex * dx1 - ex * dx2 - ex * dx3 + ex * dx4
    outVec[1] = - ex * dx1 - ex * dx2 + ex * dx3 + ex * dx4
    outVec[2] = - ex * dx1 + ex * dx2 - ex * dx3 + ex * dx4

    normalize(outVec, outVec)



@cuda.jit('float64(float64, float64, float64)', device=True)
def mix(xx, yy, aa):
    return xx * (1-aa) + yy * aa


# http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
@cuda.jit('float64(float64[:], float64[:], float64, float64, float64[:])', device=True)
def calcSoftshadow(rayOrigin, rayDirection, mint, tmax, attributes):
    result = cuda.local.array(shape=3, dtype=float64)
    position = cuda.local.array(shape=3, dtype=float64)

    res = 1.0
    t = mint
    for i in range(16):
        position[0] = rayOrigin[0] + rayDirection[0] * t
        position[1] = rayOrigin[1] + rayDirection[1] * t
        position[2] = rayOrigin[2] + rayDirection[2] * t
        worldMap( result, position, attributes )
        h = result[0]
        res = min( res, 8.0 * h / t )
        t += clip( h, 0.02, 0.10 )
        if res < 0.005 or t > tmax: 
            break
    return clip( res, 0.0, 1.0 )




@cuda.jit('float64(float64[:], float64[:], float64[:])', device=True)
def calcAO(position, normal, attributes):
    result = cuda.local.array(shape=3, dtype=float64)
    aopos = cuda.local.array(shape=3, dtype=float64)

    occ = 0.0
    sca = 1.0
    for i in range(5):
        hr = 0.01 + 0.12 * float64(i) / 4.0
        aopos[0] = position[0] + normal[0] * hr
        aopos[1] = position[1] + normal[1] * hr
        aopos[2] = position[2] + normal[2] * hr
        worldMap(result, aopos, attributes)
        dd = result[0]
        occ += -(dd-hr) * sca
        sca *= 0.95

    return clip(1.0 - 3.0 * occ, 0.0, 1.0)




@cuda.jit('void(float64[:], float64, float64, float64[:,:], float64[:,:], float64[:,:])', device=True)
def drawDCT(color, XX, YY, FR, FG, FB):
    
    currentCol = cuda.local.array(shape=3, dtype=float64)

    color[0] = 0.0
    color[1] = 0.0
    color[2] = 0.0

    for uu in range(3):
        for vv in range(3):

            currentCol[0] = FR[uu, vv] * math.cos((2.0 * XX + 1.0) * float64(uu) * math.pi / 16.0) * math.cos((2.0 * YY + 1.0) * float64(vv) * math.pi / 16.0)
            currentCol[1] = FG[uu, vv] * math.cos((2.0 * XX + 1.0) * float64(uu) * math.pi / 16.0) * math.cos((2.0 * YY + 1.0) * float64(vv) * math.pi / 16.0)
            currentCol[2] = FB[uu, vv] * math.cos((2.0 * XX + 1.0) * float64(uu) * math.pi / 16.0) * math.cos((2.0 * YY + 1.0) * float64(vv) * math.pi / 16.0)

            if vv == 0:
                currentCol[0] = currentCol[0] * 0.707107 
                currentCol[1] = currentCol[1] * 0.707107 
                currentCol[2] = currentCol[2] * 0.707107 
            if uu == 0:
                currentCol[0] = currentCol[0] * 0.707107 
                currentCol[1] = currentCol[1] * 0.707107 
                currentCol[2] = currentCol[2] * 0.707107 

            color[0] = color[0] + currentCol[0]
            color[1] = color[1] + currentCol[1]
            color[2] = color[2] + currentCol[2]

    color[0] = color[0] / 4.0
    color[1] = color[1] / 4.0
    color[2] = color[2] / 4.0


@cuda.jit('void(float64[:], float64[:], float64[:], float64[:], float64[:])', device=True)
def render(color, rayOrigin, rayDirection, result, attributes):
    
    position = cuda.local.array(shape=3, dtype=float64)
    normal = cuda.local.array(shape=3, dtype=float64)
    lin = cuda.local.array(shape=3, dtype=float64)
    light = cuda.local.array(shape=3, dtype=float64)
    reflection = cuda.local.array(shape=3, dtype=float64)
    hal = cuda.local.array(shape=3, dtype=float64)

    #color[0] = 0.7 + rayDirection[1] * 0.8
    #color[1] = 0.9 + rayDirection[1] * 0.8
    #color[2] = 1.0 + rayDirection[1] * 0.8
    
    castRay(result, rayOrigin, rayDirection, attributes)

    if result[1] > -0.5:

        color[0] = attributes[8]
        color[1] = attributes[9]
        color[2] = attributes[10]
        
        position[0] = rayOrigin[0] + result[0]*rayDirection[0]
        position[1] = rayOrigin[1] + result[0]*rayDirection[1]
        position[2] = rayOrigin[2] + result[0]*rayDirection[2]
        calcNormal(normal, position, attributes)
        reflect(reflection, rayDirection, normal)
        
        # lighitng        
        occ = calcAO( position, normal, attributes )

        light[0] = -0.4
        light[1] = 0.7
        light[2] = -0.6
        normalize(light, light)

        amb = clip( 0.5 + 0.5*normal[1], 0.0, 1.0 )
        dif = clip( dotProduct( normal, light ), 0.0, 1.0 )
        
        dif *= calcSoftshadow( position, light, 0.02, 2.5, attributes )
        
        hal[0] = light[0] - rayDirection[0]
        hal[1] = light[1] - rayDirection[1]
        hal[2] = light[2] - rayDirection[2]
        normalize(hal, hal)
        spe = pow( clip( dotProduct( normal, hal ), 0.0, 1.0 ), 16.0) * dif * (0.04 + 0.96 * pow( clip(1.0+dotProduct(hal,rayDirection),0.0,1.0), 5.0 ))
       
        hal[0] = -light[0]
        hal[1] = 0.0
        hal[2] = -light[2]
        normalize(hal, hal)
        bac = clip(dotProduct(normal, hal), 0.0, 1.0 )*clip( 1.0-position[1],0.0,1.0)
        dom = smoothstep( -0.2, 0.2, reflection[1] )
        fre = pow( clip(1.0+dotProduct(normal,rayDirection),0.0,1.0), 2.0 )

        dom *= calcSoftshadow( position, reflection, 0.02, 2.5, attributes )


        lin[0] = 1.30 * dif * 1.00
        lin[1] = 1.30 * dif * 0.80
        lin[2] = 1.30 * dif * 0.55

        lin[0] += 0.30 * amb * 0.40 * occ
        lin[1] += 0.30 * amb * 0.60 * occ
        lin[2] += 0.30 * amb * 1.00 * occ

        lin[0] += 0.40 * dom * 0.40 * occ
        lin[1] += 0.40 * dom * 0.60 * occ
        lin[2] += 0.40 * dom * 1.00 * occ

        lin[0] += 0.50 * bac * 0.25 * occ
        lin[1] += 0.50 * bac * 0.25 * occ
        lin[2] += 0.50 * bac * 0.25 * occ

        lin[0] += 0.25 * fre * 1.00 * occ
        lin[1] += 0.25 * fre * 1.00 * occ
        lin[2] += 0.25 * fre * 1.00 * occ
        
        color[0] = color[0] * lin[0]
        color[1] = color[1] * lin[1]
        color[2] = color[2] * lin[2]

        color[0] += 9.0 * spe * 1.0
        color[1] += 9.0 * spe * 0.9
        color[2] += 9.0 * spe * 0.7

        dx = 1.0-math.exp( -0.0002*result[0]*result[0]*result[0] )
        color[0] = mix( color[0], 0.8, dx )
        color[1] = mix( color[1], 0.9, dx )
        color[2] = mix( color[2], 1.0, dx )

    clipVec(color, 0.0, 1.0)



@cuda.jit
def mainRender(io_array, width, height, attributes):
    offset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    miscVec1 = cuda.local.array(shape=3, dtype=float64)
    miscVec2 = cuda.local.array(shape=3, dtype=float64)
    miscVec3 = cuda.local.array(shape=3, dtype=float64)

    totalColor = cuda.local.array(shape=3, dtype=float64)
    color = cuda.local.array(shape=3, dtype=float64)
    rayOrigin = cuda.local.array(shape=3, dtype=float64)
    rayDirection = cuda.local.array(shape=3, dtype=float64)
    rayDirection = cuda.local.array(shape=3, dtype=float64)
    cameraToWorld = cuda.local.array(shape=(3,3), dtype=float64)
    resultVec = cuda.local.array(shape=2, dtype=float64)

    totalColor[0] = 0.0
    totalColor[1] = 0.0
    totalColor[2] = 0.0
    color[0] = 0.0
    color[1] = 0.0
    color[2] = 0.0

    AA = 2

    xx = offset / height
    yy = offset % height    
    
    FR = cuda.local.array(shape=(3,3), dtype=float64)
    FG = cuda.local.array(shape=(3,3), dtype=float64)
    FB = cuda.local.array(shape=(3,3), dtype=float64)

    FR[0,0] = attributes[11] * 1.0
    FR[0,1] = attributes[12] * 1.0
    FR[0,2] = attributes[13] * 1.0
    FR[1,0] = attributes[14] * 1.0
    FR[1,1] = attributes[15] * 1.0
    FR[1,2] = attributes[16] * 1.0
    FR[2,0] = attributes[17] * 1.0
    FR[2,1] = attributes[18] * 1.0
    FR[2,2] = attributes[19] * 1.0
    
    FG[0,0] = attributes[20] * 1.0
    FG[0,1] = attributes[21] * 1.0
    FG[0,2] = attributes[22] * 1.0
    FG[1,0] = attributes[23] * 1.0
    FG[1,1] = attributes[24] * 1.0
    FG[1,2] = attributes[25] * 1.0
    FG[2,0] = attributes[26] * 1.0
    FG[2,1] = attributes[27] * 1.0
    FG[2,2] = attributes[28] * 1.0

    FB[0,0] = attributes[29] * 1.0
    FB[0,1] = attributes[30] * 1.0
    FB[0,2] = attributes[31] * 1.0
    FB[1,0] = attributes[32] * 1.0
    FB[1,1] = attributes[33] * 1.0
    FB[1,2] = attributes[34] * 1.0
    FB[2,0] = attributes[35] * 1.0
    FB[2,1] = attributes[36] * 1.0
    FB[2,2] = attributes[37] * 1.0
    
    for m in range(AA):
        for n in range(AA):
            # pixel coordinates
            miscVec1[0] = (float64(m) / float64(AA)) - 0.5
            miscVec1[1]  = (float64(n) / float64(AA)) - 0.5

            miscVec1[0] = (2.0 * (xx + miscVec1[0]) - float64(width)) / float64(height)
            miscVec1[1] = (2.0 * (yy + miscVec1[1]) - float64(height)) / float64(height)
            miscVec1[2] = 4.0
            normalize(miscVec1, miscVec1)

            # camera position
            rayOrigin[0] = 0.0
            rayOrigin[1] = 2.5
            rayOrigin[2] = -1.5
            
            # camera look at
            miscVec2[0] = 0.0
            miscVec2[1] = 1.0
            miscVec2[2] = 0.0
            
            # camera-to-world transformation
            setCamera(cameraToWorld, rayOrigin, miscVec2, 0.0, miscVec3)

            # ray direction
            matMul(rayDirection, cameraToWorld, miscVec1)

            # DCT Colors
            drawDCT(color, float64(xx * 8) / float64(width), float64(yy * 8) / float64(height), FR, FG, FB)

            # render    
            render(color, rayOrigin, rayDirection, resultVec, attributes)

            # gamma
            totalColor[0] += pow(color[0], 0.4545)
            totalColor[1] += pow(color[1], 0.4545)
            totalColor[2] += pow(color[2], 0.4545)
    

    for i in range(3):
        io_array[offset, i] = totalColor[i] / float64(AA*AA)



# Goal: Set Size + Screen-to-Object ratio

tWidth = 28 #600
tHeight = 28 #600

# Attributes:
# 0  -> Shape
# 1  -> RotationXSin
# 2  -> RotationYSin
# 3  -> RotationZSin
# 4  -> RotationXCos
# 5  -> RotationYCos
# 6  -> RotationZCos
# 7  -> Size
# 8  -> ColorR
# 9  -> ColorG
# 10 -> ColorB
# 11 -> DCTR[0,0]
# 12 -> DCTR[0,1]
# 13 -> DCTR[0,2]
# 14 -> DCTR[1,0]
# 15 -> DCTR[1,1]
# 16 -> DCTR[1,2]
# 17 -> DCTR[2,0]
# 18 -> DCTR[2,1]
# 19 -> DCTR[2,2]
# 20 -> DCTG[0,0]
# 21 -> DCTG[0,1]
# 22 -> DCTG[0,2]
# 23 -> DCTG[1,0]
# 24 -> DCTG[1,1]
# 25 -> DCTG[1,2]
# 26 -> DCTG[2,0]
# 27 -> DCTG[2,1]
# 28 -> DCTG[2,2]
# 29 -> DCTB[0,0]
# 30 -> DCTB[0,1]
# 31 -> DCTB[0,2]
# 32 -> DCTB[1,0]
# 33 -> DCTB[1,1]
# 34 -> DCTB[1,2]
# 35 -> DCTB[2,0]
# 36 -> DCTB[2,1]
# 37 -> DCTB[2,2]

# Actual Attributes:
xAngle = 0.0
yAngle = 0.0
zAngle = 0.0

#####

mainAttributes = np.random.rand(38)

mainAttributes[0] = 1.0

mainAttributes[1] = math.sin(math.pi * 0.5 * zAngle)
mainAttributes[4] = math.cos(math.pi * 0.5 * zAngle)

mainAttributes[2] = math.sin(math.pi * 0.5 * yAngle)
mainAttributes[5] = math.cos(math.pi * 0.5 * yAngle)

mainAttributes[3] = math.sin(math.pi * 0.5 * (xAngle - 0.5))
mainAttributes[6] = math.cos(math.pi * 0.5 * (xAngle - 0.5))

mainAttributes[7] = 1.0


deviceAttributes = cuda.to_device(mainAttributes)

data = np.zeros((tWidth * tHeight, 3), dtype=float)
threadsperblock = 32 
blockspergrid = (tHeight * tWidth + (threadsperblock - 1)) // threadsperblock

for xa in range(1):
    start = time.time()
    mainRender[blockspergrid, threadsperblock](data, tWidth, tHeight, deviceAttributes)
    end = time.time()
    print("Time needed for Render: " + str(end - start))

pixels = np.zeros((tHeight, tWidth, 3), dtype=float)
for xx in range(tWidth):
    for yy in range(tHeight):
        pixels[tHeight - 1 - yy][xx] = data[xx * tHeight + yy]

newGUI = GraphicsUserInterface()

newGUI.drawArray(pixels, tWidth, tHeight)