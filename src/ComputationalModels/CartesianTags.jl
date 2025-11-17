
"Shortcuts for the tags of cartesian discrete models."
module CartesianTags

"Tags indicating points, edges and faces at plane X0."
const faceX0 = [1, 3, 5, 7, 13, 15, 17, 19, 25]

"Tags indicating points, edges and faces at plane X1."
const faceX1 = [2, 4, 6, 8, 14, 16, 18, 20, 26]

"Tags indicating points, edges and faces at plane Y0."
const faceY0 = [1, 2, 5, 6, 9, 11, 17, 18, 23]

"Tags indicating points, edges and faces at plane Y1."
const faceY1 = [3, 4, 7, 8, 10, 12, 19, 20, 24]

"Tags indicating points, edges and faces at plane Z0."
const faceZ0 = [1, 2, 3, 4, 9, 10, 13, 14, 21]

"Tags indicating points, edges and faces at plane Z1."
const faceZ1 = [5, 6, 7, 8, 11, 12, 15, 16, 22]


"Tag indicating the edge at  X, Y0, Z0."
const edgeX00 = [9]

"Tag indicating the edge at  X, Y1, Z0."
const edgeX10 = [10]

"Tag indicating the edge at  X, Y0, Z1."
const edgeX01 = [11]

"Tag indicating the edge at  X, Y1, Z1."
const edgeX11 = [12]

"Tag indicating the edge at  X0, Y, Z0."
const edge0Y0 = [13]

"Tag indicating the edge at  X1, Y, Z0."
const edge1Y0 = [14]

"Tag indicating the edge at  X0, Y, Z1."
const edge0Y1 = [15]

"Tag indicating the edge at  X1, Y, Z1."
const edge1Y1 = [16]

"Tag indicating the edge at  X0, Y0, Z."
const edge00Z = [17]

"Tag indicating the edge at  X1, Y0, Z."
const edge10Z = [18]

"Tag indicating the edge at  X0, Y1, Z."
const edge01Z = [19]

"Tag indicating the edge at  X1, Y1, Z."
const edge11Z = [20]


"Tag indicating the point at corner X0, Y0, Z0."
const corner000 = [1]

"Tag indicating the point at corner X1, Y0, Z0."
const corner100 = [2]

"Tag indicating the point at corner X0, Y1, Z0."
const corner010 = [3]

"Tag indicating the point at corner X1, Y1, Z0."
const corner110 = [4]

"Tag indicating the point at corner X0, Y0, Z1."
const corner001 = [5]

"Tag indicating the point at corner X1, Y0, Z1."
const corner101 = [6]

"Tag indicating the point at corner X0, Y1, Z1."
const corner011 = [7]

"Tag indicating the point at corner X1, Y1, Z1."
const corner111 = [8]
end
