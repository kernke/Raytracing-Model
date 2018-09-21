# Monte-Carlo-Raytracing

Ray tracing is an algorithm for the generation of photo-realistic computer 
	graphics. Starting with a 3-dimensional object and the positions of the 
	light source and the camera, the algorithm computes the 2-dimensional 
	picture of the object according to the given lighting. By tracing a light 
	ray not only for the straight line distance between two surfaces, but 
	instead tracking it further, 
	reflections and refraction need to be taken into account. So a ray 
	from the sun could first be reflected by a building and then reach the 
	roof of a car and then fall into the camera.
	Thus in computer graphics shiny surfaces appear more realistic, showing for 
	example	the reflection of the sun or other objects on a surface. To avoid 
	calculating 
	rays, that do not reach the camera, usually the tracing is done 
	"backwards". Meaning, that starting from the camera the rays are 
	projected on the object. The different pixels of the picture are 
	represented by different angles deviating from the direction the camera is 
	pointing at, thereby forming an opening cone, which is extended to the 
	objects surface. 
	Executing the principle of ray tracing rigorously the full path from the 
	camera to the light source 
	via reflections and refractions on the object would be needed.

But often there is an interest only in the aesthetics of the picture and 
	the tracing can be stopped after 2 or 3 reflections and some heuristic 
	surface 
	brightness value can be used from there on. 
	Since many surfaces are not highly reflective or smooth, the amount of 
	light 
	by diffuse reflections makes geometric reflections of higher order 
	negligible and justifies this approach.
	
The method of raytracing can also be used in a scientific way to analyze 
	images. The algorithm generates a picture, given the object and lighting. 
	If the image already exists from an experimental setup, one can use 
	ray tracing to inversely reconstruct either: 
		
		the lighting, given the structure of the object or
		
		the structure of the object, given the setup of the lighting.

Now one can try different hypotheses about the unknown part, and compare 
	the computed image with the experimental result.
	Even if the reproduction of an image by ray tracing is successful, the 
	hypothesis needs additional reasoning, because the solution does not have 
	to be unique.  

	

	
