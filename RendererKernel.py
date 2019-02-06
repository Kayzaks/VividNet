import math
from numba import cuda, float32, int32


@cuda.jit('void(float32[:], float32[:], float32[:])', device=True)
def crossProduct(vecOut, vec1, vec2):
    vecOut[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    vecOut[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    vecOut[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]


@cuda.jit('float32(float32[:], float32[:])', device=True)
def dotProduct(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]
    
@cuda.jit('float32(float32[:])', device=True)
def vecNorm(vec1):
    return math.sqrt(vec1[0] * vec1[0] + vec1[1] * vec1[1] + vec1[2] * vec1[2])

@cuda.jit('void(float32[:], float32[:], float32[:])', device=True)
def reflect(outVec, incident, normal):
    dx = dotProduct(incident, normal)
    outVec[0] = incident[0] - 2.0 * dx * normal[0]
    outVec[1] = incident[1] - 2.0 * dx * normal[1]
    outVec[2] = incident[2] - 2.0 * dx * normal[2]

@cuda.jit('void(float32[:], float32[:])', device=True)
def normalize(vecOut, vec1):
    dist = vecNorm(vec1)
    vecOut[0] = vec1[0] / dist
    vecOut[1] = vec1[1] / dist
    vecOut[2] = vec1[2] / dist
    
@cuda.jit('void(float32[:], float32[:, :], float32[:])', device=True)
def matMul(vecOut, matrix, vec):
    vecOut[0] = matrix[0, 0] * vec[0] + matrix[1, 0] * vec[1] + matrix[2, 0] * vec[2] 
    vecOut[1] = matrix[0, 1] * vec[0] + matrix[1, 1] * vec[1] + matrix[2, 1] * vec[2] 
    vecOut[2] = matrix[0, 2] * vec[0] + matrix[1, 2] * vec[1] + matrix[2, 2] * vec[2] 


@cuda.jit('void(float32[:, :], float32[:], float32[:], float32, float32[:])', device=True)
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


@cuda.jit('void(float32[:], float32, float32)', device=True)
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
        
@cuda.jit('float32(float32, float32, float32)', device=True)
def clip(value, leftBound, rightBound):
    if value > rightBound:
        return rightBound
    elif value < leftBound:
        return leftBound
    return value


@cuda.jit('float32(float32, float32, float32)', device=True)
def smoothstep(edge0, edge1, x):
    t = clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)



@cuda.jit('float32(float32[:])', device=True)
def sdPlane(point):
    return point[1]

@cuda.jit('float32(float32[:], float32)', device=True)
def sdSphere(point, radius):
    return vecNorm(point) - radius


@cuda.jit('float32(float32[:], float32, float32, float32)', device=True)
def sdBox(point, boundX, boundY, boundZ):
    point[0] = abs(point[0]) - boundX
    point[1] = abs(point[1]) - boundY
    point[2] = abs(point[2]) - boundZ
    outVal = min(max(point[0],max(point[1],point[2])),0.0)
    point[0] = max(point[0], 0.0)
    point[1] = max(point[1], 0.0)
    point[2] = max(point[2], 0.0)
    return outVal + vecNorm(point)



@cuda.jit('void(float32[:], float32[:], float32[:])', device=True)
def opU(result, dist1, dist2):
    if dist1[0] < dist2[0]:
        result[0] = dist1[0]
        result[1] = dist1[1]
    else:
        result[0] = dist2[0]
        result[1] = dist2[1]


@cuda.jit('void(float32[:], float32[:], float32[:])', device=True)
def worldMap(result, position, attributes):
    vec3d = cuda.local.array(shape=3, dtype=float32)

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

    if attributes[0] < 0.5:
        result[0] = sdBox(vec3d, 0.3 * attributes[7], 0.3 * attributes[8], 0.3 * attributes[9])
    elif attributes[0] < 1.5:
        result[0] = sdSphere(vec3d, 0.5 * attributes[7])

    result[1] = 2.0

    


@cuda.jit('void(float32[:], float32[:], float32[:], float32[:])', device=True)
def castRay(result, rayOrigin, rayDirection, attributes):
    tmin = 1.0
    tmax = 20.0
    
    pos = cuda.local.array(shape=3, dtype=float32)

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
@cuda.jit('void(float32[:], float32[:], float32[:])', device=True)
def calcNormal(outVec, position, attributes):
    auxVec = cuda.local.array(shape=3, dtype=float32)

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



@cuda.jit('float32(float32, float32, float32)', device=True)
def mix(xx, yy, aa):
    return xx * (1-aa) + yy * aa


# http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
@cuda.jit('float32(float32[:], float32[:], float32, float32, float32[:])', device=True)
def calcSoftshadow(rayOrigin, rayDirection, mint, tmax, attributes):
    result = cuda.local.array(shape=3, dtype=float32)
    position = cuda.local.array(shape=3, dtype=float32)

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




@cuda.jit('float32(float32[:], float32[:], float32[:])', device=True)
def calcAO(position, normal, attributes):
    result = cuda.local.array(shape=3, dtype=float32)
    aopos = cuda.local.array(shape=3, dtype=float32)

    occ = 0.0
    sca = 1.0
    for i in range(5):
        hr = 0.01 + 0.12 * float32(i) / 4.0
        aopos[0] = position[0] + normal[0] * hr
        aopos[1] = position[1] + normal[1] * hr
        aopos[2] = position[2] + normal[2] * hr
        worldMap(result, aopos, attributes)
        dd = result[0]
        occ += -(dd-hr) * sca
        sca *= 0.95

    return clip(1.0 - 3.0 * occ, 0.0, 1.0)




@cuda.jit('void(float32[:], float32, float32, float32[:,:], float32[:,:], float32[:,:])', device=True)
def drawDCT(color, XX, YY, FR, FG, FB):
    
    currentCol = cuda.local.array(shape=3, dtype=float32)

    color[0] = 0.0
    color[1] = 0.0
    color[2] = 0.0

    for uu in range(3):
        for vv in range(3):

            currentCol[0] = FR[uu, vv] * math.cos((2.0 * XX + 1.0) * float32(uu) * math.pi / 16.0) * math.cos((2.0 * YY + 1.0) * float32(vv) * math.pi / 16.0)
            currentCol[1] = FG[uu, vv] * math.cos((2.0 * XX + 1.0) * float32(uu) * math.pi / 16.0) * math.cos((2.0 * YY + 1.0) * float32(vv) * math.pi / 16.0)
            currentCol[2] = FB[uu, vv] * math.cos((2.0 * XX + 1.0) * float32(uu) * math.pi / 16.0) * math.cos((2.0 * YY + 1.0) * float32(vv) * math.pi / 16.0)

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


@cuda.jit('void(float32[:], float32[:], float32[:], float32[:], float32[:])', device=True)
def render(color, rayOrigin, rayDirection, result, attributes):
    
    position = cuda.local.array(shape=3, dtype=float32)
    normal = cuda.local.array(shape=3, dtype=float32)
    lin = cuda.local.array(shape=3, dtype=float32)
    light = cuda.local.array(shape=3, dtype=float32)
    reflection = cuda.local.array(shape=3, dtype=float32)
    halfAngle = cuda.local.array(shape=3, dtype=float32)
    
    castRay(result, rayOrigin, rayDirection, attributes)

    if result[1] > -0.5:

        color[0] = attributes[46]
        color[1] = attributes[47]
        color[2] = attributes[48]
        
        position[0] = rayOrigin[0] + result[0]*rayDirection[0]
        position[1] = rayOrigin[1] + result[0]*rayDirection[1]
        position[2] = rayOrigin[2] + result[0]*rayDirection[2]
        calcNormal(normal, position, attributes)
        reflect(reflection, rayDirection, normal)
             
        # Ambient Occlusion
        occ = calcAO( position, normal, attributes )

        ## Lighting 
        light[0] = attributes[43]
        light[1] = attributes[44]
        light[2] = attributes[45]

        # Ambient
        ambient = clip( 0.5 + 0.5*normal[1], 0.0, 1.0 )

        # Diffuse - Lambertian
        diffuse = clip( dotProduct( normal, light ), 0.0, 1.0 )
        diffuse *= calcSoftshadow( position, light, 0.02, 2.5, attributes )
        
        # Specular - Blinn-Phong
        halfAngle[0] = light[0] - rayDirection[0]
        halfAngle[1] = light[1] - rayDirection[1]
        halfAngle[2] = light[2] - rayDirection[2]
        normalize(halfAngle, halfAngle)
        specular = pow( clip( dotProduct( normal, halfAngle ), 0.0, 1.0 ), attributes[53]) # * diffuse * (0.04 + 0.96 * pow( clip(1.0+dotProduct(halfAngle, rayDirection),0.0,1.0), 25.0 ))
       
        # Fresnel
        halfAngle[0] = -light[0]
        halfAngle[1] = 0.0
        halfAngle[2] = -light[2]
        normalize(halfAngle, halfAngle)
        fresnel = pow( clip(1.0+dotProduct(normal,rayDirection),0.0,1.0), attributes[58] )

        #        Diffuse Power              Diffuse Color    Light Color
        lin[0] = attributes[49] * diffuse * attributes[50] * attributes[40]
        lin[1] = attributes[49] * diffuse * attributes[51] * attributes[41]
        lin[2] = attributes[49] * diffuse * attributes[52] * attributes[42]

        
        #         Ambient Power              Ambient Color
        lin[0] += attributes[59] * ambient * attributes[60] * occ
        lin[1] += attributes[59] * ambient * attributes[61] * occ
        lin[2] += attributes[59] * ambient * attributes[62] * occ

        lin[0] += 0.25 * fresnel * 1.00 * occ
        lin[1] += 0.25 * fresnel * 1.00 * occ
        lin[2] += 0.25 * fresnel * 1.00 * occ
        
        color[0] = color[0] * lin[0]
        color[1] = color[1] * lin[1]
        color[2] = color[2] * lin[2]
        #           Specular Power              Specular Color   Light Color
        color[0] += attributes[54] * specular * attributes[55] * attributes[40] 
        color[1] += attributes[54] * specular * attributes[56] * attributes[41]
        color[2] += attributes[54] * specular * attributes[57] * attributes[42]

        dx = 1.0-math.exp( -0.0002*result[0]*result[0]*result[0] )
        color[0] = mix( color[0], 0.8, dx )
        color[1] = mix( color[1], 0.9, dx )
        color[2] = mix( color[2], 1.0, dx )


    clipVec(color, 0.0, 1.0)

    # Depth
    color[3] = result[0] 



@cuda.jit
def mainRender(io_array, width, height, attributes):
    offset = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    miscVec1 = cuda.local.array(shape=3, dtype=float32)
    miscVec2 = cuda.local.array(shape=3, dtype=float32)
    miscVec3 = cuda.local.array(shape=3, dtype=float32)

    totalColor = cuda.local.array(shape=3, dtype=float32)
    color = cuda.local.array(shape=3, dtype=float32)
    rayOrigin = cuda.local.array(shape=3, dtype=float32)
    rayDirection = cuda.local.array(shape=3, dtype=float32)
    rayDirection = cuda.local.array(shape=3, dtype=float32)
    cameraToWorld = cuda.local.array(shape=(3,3), dtype=float32)
    resultVec = cuda.local.array(shape=2, dtype=float32)
    FR = cuda.local.array(shape=(3,3), dtype=float32)
    FG = cuda.local.array(shape=(3,3), dtype=float32)
    FB = cuda.local.array(shape=(3,3), dtype=float32)

    totalColor[0] = 0.0
    totalColor[1] = 0.0
    totalColor[2] = 0.0

    AA = 2

    xx = offset / height
    yy = offset % height    
    
    FR[0,0] = attributes[13] * 1.0
    FR[0,1] = attributes[14] * 1.0
    FR[0,2] = attributes[15] * 1.0
    FR[1,0] = attributes[16] * 1.0
    FR[1,1] = attributes[17] * 1.0
    FR[1,2] = attributes[18] * 1.0
    FR[2,0] = attributes[19] * 1.0
    FR[2,1] = attributes[20] * 1.0
    FR[2,2] = attributes[21] * 1.0
    
    FG[0,0] = attributes[22] * 1.0
    FG[0,1] = attributes[23] * 1.0
    FG[0,2] = attributes[24] * 1.0
    FG[1,0] = attributes[25] * 1.0
    FG[1,1] = attributes[26] * 1.0
    FG[1,2] = attributes[27] * 1.0
    FG[2,0] = attributes[28] * 1.0
    FG[2,1] = attributes[29] * 1.0
    FG[2,2] = attributes[30] * 1.0

    FB[0,0] = attributes[31] * 1.0
    FB[0,1] = attributes[32] * 1.0
    FB[0,2] = attributes[33] * 1.0
    FB[1,0] = attributes[34] * 1.0
    FB[1,1] = attributes[35] * 1.0
    FB[1,2] = attributes[36] * 1.0
    FB[2,0] = attributes[37] * 1.0
    FB[2,1] = attributes[38] * 1.0
    FB[2,2] = attributes[39] * 1.0
    
    for m in range(AA):
        for n in range(AA):
            
            color[0] = 0.0
            color[1] = 0.0
            color[2] = 0.0
            color[3] = 1000.0
            
            # pixel coordinates
            miscVec1[0] = (float32(m) / float32(AA)) - 0.5
            miscVec1[1]  = (float32(n) / float32(AA)) - 0.5

            miscVec1[0] = (2.0 * (xx + miscVec1[0]) - float32(width)) / float32(height)
            miscVec1[1] = (2.0 * (yy + miscVec1[1]) - float32(height)) / float32(height)
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
            drawDCT(color, float32(xx * 8) / float32(width), float32(yy * 8) / float32(height), FR, FG, FB)

            # render    
            render(color, rayOrigin, rayDirection, resultVec, attributes)

            # gamma
            totalColor[0] += pow(color[0], 0.4545)
            totalColor[1] += pow(color[1], 0.4545)
            totalColor[2] += pow(color[2], 0.4545)
    

    for i in range(3):
        io_array[offset, i] = totalColor[i] / float32(AA*AA)
    
    # Depth
    io_array[offset, 3] = color[3] 


