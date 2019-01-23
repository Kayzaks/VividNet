
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


@cuda.jit('void(float64[:], float64[:])', device=True)
def worldMap(result, position):
    vec2d1 = cuda.local.array(shape=2, dtype=float64)
    vec2d2 = cuda.local.array(shape=2, dtype=float64)
    vec3d = cuda.local.array(shape=3, dtype=float64)

    vec2d1[0] = sdPlane(position)
    vec2d1[1] = 1.0

    vec3d[0] = position[0] - 0.0
    vec3d[1] = position[1] - 0.25
    vec3d[2] = position[2] - 0.0
    
    vec2d2[0] = sdSphere(vec3d, 0.25)
    vec2d2[1] = 2.0

    opU(result, vec2d1, vec2d2)

    vec3d[0] = position[0] - 1.0
    vec3d[1] = position[1] - 0.25
    vec3d[2] = position[2] - 0.0
    
    vec2d2[0] = sdBox(vec3d, 0.25, 0.25, 0.25)
    vec2d2[1] = 3.0

    opU(result, result, vec2d2)
    


@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def castRay(result, rayOrigin, rayDirection):
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

        worldMap(result, pos)

        if result[0] < precis or t > tmax:
            break
        t += result[0]
        m = result[1]

    if t > tmax: 
        m=-1.0
    
    result[0] = t
    result[1] = m

# http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
@cuda.jit('void(float64[:], float64[:])', device=True)
def calcNormal(outVec, position):
    auxVec = cuda.local.array(shape=3, dtype=float64)

    ex = 0.5773 * 0.0005

    auxVec[0] = position[0] + ex
    auxVec[1] = position[1] - ex
    auxVec[2] = position[2] - ex
    worldMap(outVec, auxVec)
    dx1 = outVec[0]

    auxVec[0] = position[0] - ex
    auxVec[1] = position[1] - ex
    auxVec[2] = position[2] + ex
    worldMap(outVec, auxVec)
    dx2 = outVec[0]

    auxVec[0] = position[0] - ex
    auxVec[1] = position[1] + ex
    auxVec[2] = position[2] - ex
    worldMap(outVec, auxVec)
    dx3 = outVec[0]

    auxVec[0] = position[0] + ex
    auxVec[1] = position[1] + ex
    auxVec[2] = position[2] + ex
    worldMap(outVec, auxVec)
    dx4 = outVec[0]

    outVec[0] =   ex * dx1 - ex * dx2 - ex * dx3 + ex * dx4
    outVec[1] = - ex * dx1 - ex * dx2 + ex * dx3 + ex * dx4
    outVec[2] = - ex * dx1 + ex * dx2 - ex * dx3 + ex * dx4

    normalize(outVec, outVec)



@cuda.jit('float64(float64, float64, float64)', device=True)
def mix(xx, yy, aa):
    return xx * (1-aa) + yy * aa


# http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
@cuda.jit('float64(float64[:], float64[:], float64, float64)', device=True)
def calcSoftshadow(rayOrigin, rayDirection, mint, tmax):
    result = cuda.local.array(shape=3, dtype=float64)
    position = cuda.local.array(shape=3, dtype=float64)

    res = 1.0
    t = mint
    for i in range(16):
        position[0] = rayOrigin[0] + rayDirection[0] * t
        position[1] = rayOrigin[1] + rayDirection[1] * t
        position[2] = rayOrigin[2] + rayDirection[2] * t
        worldMap( result, position )
        h = result[0]
        res = min( res, 8.0 * h / t )
        t += clip( h, 0.02, 0.10 )
        if res < 0.005 or t > tmax: 
            break
    return clip( res, 0.0, 1.0 )




@cuda.jit('float64(float64[:], float64[:])', device=True)
def calcAO(position, normal):
    result = cuda.local.array(shape=3, dtype=float64)
    aopos = cuda.local.array(shape=3, dtype=float64)

    occ = 0.0
    sca = 1.0
    for i in range(5):
        hr = 0.01 + 0.12 * float64(i) / 4.0
        aopos[0] = position[0] + normal[0] * hr
        aopos[1] = position[1] + normal[1] * hr
        aopos[2] = position[2] + normal[2] * hr
        worldMap(result, aopos)
        dd = result[0]
        occ += -(dd-hr) * sca
        sca *= 0.95

    return clip(1.0 - 3.0 * occ, 0.0, 1.0)



@cuda.jit('void(float64[:], float64[:], float64[:], float64[:])', device=True)
def render(color, rayOrigin, rayDirection, result):
    
    position = cuda.local.array(shape=3, dtype=float64)
    normal = cuda.local.array(shape=3, dtype=float64)
    lin = cuda.local.array(shape=3, dtype=float64)
    light = cuda.local.array(shape=3, dtype=float64)
    reflection = cuda.local.array(shape=3, dtype=float64)
    hal = cuda.local.array(shape=3, dtype=float64)

    color[0] = 0.7 + rayDirection[1] * 0.8
    color[1] = 0.9 + rayDirection[1] * 0.8
    color[2] = 1.0 + rayDirection[1] * 0.8

    castRay(result, rayOrigin, rayDirection)

    if result[1] > -0.5:

        if result[1] < 1.5:
            # Floor Color
            color[0] = 0.3
            color[1] = 0.3
            color[2] = 0.3
        elif result[1] < 2.5:
            # Sphere Color
            color[0] = 0.8
            color[1] = 0.1
            color[2] = 0.1
        else:
            # Box Color
            color[0] = 0.8
            color[1] = 0.1
            color[2] = 0.1
        
        position[0] = rayOrigin[0] + result[0]*rayDirection[0]
        position[1] = rayOrigin[1] + result[0]*rayDirection[1]
        position[2] = rayOrigin[2] + result[0]*rayDirection[2]
        calcNormal(normal, position)
        reflect(reflection, rayDirection, normal)
        
        # lighitng        
        occ = calcAO( position, normal )

        light[0] = -0.4
        light[1] = 0.7
        light[2] = -0.6
        normalize(light, light)

        amb = clip( 0.5 + 0.5*normal[1], 0.0, 1.0 )
        dif = clip( dotProduct( normal, light ), 0.0, 1.0 )
        
        dif *= calcSoftshadow( position, light, 0.02, 2.5 )
        
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

        dom *= calcSoftshadow( position, reflection, 0.02, 2.5 )


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
def mainRender(io_array, width, height):
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
    currenTime = -10

    xx = offset / height
    yy = offset % height
    
    for m in range(AA):
        for n in range(AA):
            # pixel coordinates
            miscVec1[0] = (float64(m) / float64(AA)) - 0.5
            miscVec1[1]  = (float64(n) / float64(AA)) - 0.5

            miscVec1[0] = (2.0 * (xx + miscVec1[0]) - float64(width)) / float64(height)
            miscVec1[1] = (2.0 * (yy + miscVec1[1]) - float64(height)) / float64(height)
            miscVec1[2] = 2.0
            normalize(miscVec1, miscVec1)

            # camera    
            
            rayOrigin[0] = 4.6 * math.cos(0.1*currenTime)
            rayOrigin[1] = 1.0
            rayOrigin[2] = 0.5 + 4.6 * math.sin(0.1*currenTime)
            
            miscVec2[0] = -0.5
            miscVec2[1] = -0.4
            miscVec2[2] = 0.5
            
            # camera-to-world transformation
            setCamera(cameraToWorld, rayOrigin, miscVec2, 0.0, miscVec3)

            # ray direction
            matMul(rayDirection, cameraToWorld, miscVec1)


            # render    
            render(color, rayOrigin, rayDirection, resultVec)

            # gamma
            totalColor[0] += pow(color[0], 0.4545)
            totalColor[1] += pow(color[1], 0.4545)
            totalColor[2] += pow(color[2], 0.4545)
    

    for i in range(3):
        io_array[offset, i] = totalColor[i] / float64(AA*AA)



tWidth = 800
tHeight = 600

data = np.zeros((tWidth * tHeight, 3), dtype=float)
threadsperblock = 32 
blockspergrid = (tHeight * tWidth + (threadsperblock - 1)) // threadsperblock

for xa in range(10):
    start = time.time()

    mainRender[blockspergrid, threadsperblock](data, tWidth, tHeight)

    end = time.time()

    print("Time needed for Render: " + str(end - start))

pixels = np.zeros((tHeight, tWidth, 3), dtype=float)
for xx in range(tWidth):
    for yy in range(tHeight):
        pixels[tHeight - 1 - yy][xx] = data[xx * tHeight + yy]

newGUI = GraphicsUserInterface()

newGUI.drawArray(pixels, tWidth, tHeight)