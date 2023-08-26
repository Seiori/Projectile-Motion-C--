#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cmath>
#include "bitmap_image.hpp"
#include <chrono>
#include <iostream>

const int paperWidth = 1000;
const int paperHeight = 1000;
const int numOfParticles = 300000;
const float blendCoefficient = 0.1f;

__device__ float generateRandomFloat(float lowerBound, float upperBound, curandState* state) {
    float randomValue = curand_uniform(state);
    return lowerBound + randomValue * (upperBound - lowerBound);
}

__global__ void negate_matrix(float3* spray, int M, int N, float sprayConeAngle) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < M * N) {
        curandState state;
        curand_init(clock64(), index, 0, &state);

        // Generate Random Spray Cone Angles
        float3 sprayCone = {
            generateRandomFloat(-sprayConeAngle, sprayConeAngle, &state),
            generateRandomFloat(-sprayConeAngle, sprayConeAngle, &state),
            generateRandomFloat(-sprayConeAngle, sprayConeAngle, &state)
        };

        // Center of Paper Position
        float3 paperCenter = make_float3(paperWidth / 2.0f, 0.0f, paperHeight / 2.0f);

        // Get Direction Vector from Particle Position to Center of Paper Position
        float3 sprayDirection = {
            paperCenter.x - spray[index].x,
            paperCenter.y - spray[index].y,
            paperCenter.z - spray[index].z
        };

        // Calculate the magnitude
        float magnitude = sqrtf(sprayDirection.x * sprayDirection.x + sprayDirection.y * sprayDirection.y + sprayDirection.z * sprayDirection.z);

        // Normalize the Vector
        sprayDirection.x /= magnitude;
        sprayDirection.y /= magnitude;
        sprayDirection.z /= magnitude;

		// Convert The Spray Cone Angles to Radians
        float3 radianAngles = {
            sprayCone.x * 3.14159265358979323846f / 180.0f,
            sprayCone.y * 3.14159265358979323846f / 180.0f,
            sprayCone.z * 3.14159265358979323846f / 180.0f
        };

        // Generate The Rotation Matrix
        float cos_x = cos(radianAngles.x);
        float sin_x = sin(radianAngles.x);
        float cos_y = cos(radianAngles.y);
        float sin_y = sin(radianAngles.y);
        float cos_z = cos(radianAngles.z);
        float sin_z = sin(radianAngles.z);

        // Rotation Matrix
        float3 rotated_sprayDirection;
        rotated_sprayDirection.x = sprayDirection.x * (cos_y * cos_z) + sprayDirection.y * (cos_x * sin_z + sin_x * sin_y * cos_z) + sprayDirection.z * (sin_x * sin_z - cos_x * sin_y * cos_z);
        rotated_sprayDirection.y = -sprayDirection.x * (cos_y * sin_z) - sprayDirection.y * (cos_x * cos_z - sin_x * sin_y * sin_z) - sprayDirection.z * (sin_x * cos_z + cos_x * sin_y * sin_z);
        rotated_sprayDirection.z = sprayDirection.x * sin_y + sprayDirection.y * (-sin_x * cos_y) + sprayDirection.z * (cos_x * cos_y);

        // Initialise Variables
        float timeStep = 0.01;
        float initialVelocity = 100.0;
        float3 gravity = { 0.0f, -9.81f, 0.0f };
        float dragCoefficient = 0.025;

        float3 velocity = make_float3(rotated_sprayDirection.x * initialVelocity, rotated_sprayDirection.y * initialVelocity, rotated_sprayDirection.z * initialVelocity);

        float3 drag = { velocity.x * -dragCoefficient , velocity.y * -dragCoefficient, velocity.z * -dragCoefficient };

        float3 netAcceloration = make_float3(gravity.x + drag.x, gravity.y + drag.y, gravity.z + drag.z);

        while (spray[index].y > 0) {
            velocity.x += netAcceloration.x * timeStep;
            velocity.y += netAcceloration.y * timeStep;
            velocity.z += netAcceloration.z * timeStep;

            spray[index].x += velocity.x * timeStep;
            spray[index].y += velocity.y * timeStep;
            spray[index].z += velocity.z * timeStep;
        }
    }
}

int main() {
    // Start Timings
	auto start = std::chrono::high_resolution_clock::now();
    
    // Dynamically allocate the 2D array
    float*** paper = new float** [paperHeight];
    for (int i = 0; i < paperHeight; ++i) {
        paper[i] = new float* [paperWidth];
        for (int j = 0; j < paperWidth; ++j) {
            paper[i][j] = new float[3];
        }
    }

    // Initialize all elements of paper to (x index, 0, y index)
    for (int i = 0; i < paperHeight; i++) {
        for (int j = 0; j < paperWidth; j++) {
            paper[i][j][0] = 1.0f;
            paper[i][j][1] = 1.0f;
            paper[i][j][2] = 1.0f;
        }
    }

    // Check Elapsed Time
	auto paperInitializeEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> elapsed = paperInitializeEnd - start;
	std::cout << "Time Taken to Initialize Paper Array Elements: " << elapsed.count() << "s\n";

    // Dynamically allocate the 1D arrays
    float3* spray_red = new float3[numOfParticles];
    float3* spray_green = new float3[numOfParticles];
    float3* spray_blue = new float3[numOfParticles];

    // Initialize all elements of spray_red, spray_green, spray_blue to (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0) respectively
    for (int i = 0; i < numOfParticles; i++) {
        spray_red[i].x = -50.0f;
        spray_red[i].y = 20.0f;
        spray_red[i].z = paperHeight / 5;

		spray_green[i].x = paperWidth / 2;
        spray_green[i].y = 20.0f;
        spray_green[i].z = paperHeight + 50;

        spray_blue[i].x = paperWidth + 50;
        spray_blue[i].y = 20.0f;
        spray_blue[i].z = paperHeight / 5;
    }

    // Check Elapsed Time
	auto sprayInitializeEnd = std::chrono::high_resolution_clock::now();
	elapsed = sprayInitializeEnd - paperInitializeEnd;
	std::cout << "Time Taken to Initialize All Spray Can Elements: " << elapsed.count() << "s\n";
    
    float3* dev_spray_red;
    float3* dev_spray_green;
    float3* dev_spray_blue;

    cudaMalloc(&dev_spray_red, numOfParticles * sizeof(float3));
    cudaMalloc(&dev_spray_green, numOfParticles * sizeof(float3));
    cudaMalloc(&dev_spray_blue, numOfParticles * sizeof(float3));

    cudaMemcpy(dev_spray_red, spray_red, numOfParticles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_spray_green, spray_green, numOfParticles * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_spray_blue, spray_blue, numOfParticles * sizeof(float3), cudaMemcpyHostToDevice);

    int M = numOfParticles;
    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;

    float sprayConeAngle = 35.0f;
    negate_matrix << <numBlocks, blockSize >> > (dev_spray_red, M, 1, sprayConeAngle);
    negate_matrix << <numBlocks, blockSize >> > (dev_spray_green, M, 1, sprayConeAngle);
    negate_matrix << <numBlocks, blockSize >> > (dev_spray_blue, M, 1, sprayConeAngle);

    cudaMemcpy(spray_red, dev_spray_red, numOfParticles * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(spray_green, dev_spray_green, numOfParticles * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(spray_blue, dev_spray_blue, numOfParticles * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(dev_spray_red);
    cudaFree(dev_spray_green);
    cudaFree(dev_spray_blue);

	// Check Elapsed Time
	auto sprayConeEnd = std::chrono::high_resolution_clock::now();
	elapsed = sprayConeEnd - sprayInitializeEnd;
	std::cout << "Time Taken to Calculate Particle Movements: " << elapsed.count() << "s\n";

    int collisionCount = 0;
    
    for (int i = 0; i < numOfParticles; i++)
    {
        int roundedX_red = static_cast<int>(std::round(spray_red[i].x));
        int roundedY_red = static_cast<int>(std::round(spray_red[i].z));
        int roundedX_green = static_cast<int>(std::round(spray_green[i].x));
        int roundedY_green = static_cast<int>(std::round(spray_green[i].z));
        int roundedX_blue = static_cast<int>(std::round(spray_blue[i].x));
        int roundedY_blue = static_cast<int>(std::round(spray_blue[i].z));

        if (roundedX_red >= 0 && roundedX_red < paperWidth && roundedY_red >= 0 && roundedY_red < paperHeight)
        {
            collisionCount++;
            paper[roundedY_red][roundedX_red][0] = (1.0f - blendCoefficient) * paper[roundedY_red][roundedX_red][0] + blendCoefficient * 1.0f;
            paper[roundedY_red][roundedX_red][1] = (1.0f - blendCoefficient) * paper[roundedY_red][roundedX_red][1] + blendCoefficient * 0.0f;
            paper[roundedY_red][roundedX_red][2] = (1.0f - blendCoefficient) * paper[roundedY_red][roundedX_red][2] + blendCoefficient * 0.0f;
        }

        if (roundedX_green >= 0 && roundedX_green < paperWidth && roundedY_green >= 0 && roundedY_green < paperHeight)
        {
            collisionCount++;
            paper[roundedY_green][roundedX_green][0] = (1.0f - blendCoefficient) * paper[roundedY_green][roundedX_green][0] + blendCoefficient * 0.0f;
            paper[roundedY_green][roundedX_green][1] = (1.0f - blendCoefficient) * paper[roundedY_green][roundedX_green][1] + blendCoefficient * 1.0f;
            paper[roundedY_green][roundedX_green][2] = (1.0f - blendCoefficient) * paper[roundedY_green][roundedX_green][2] + blendCoefficient * 0.0f;
        }

        if (roundedX_blue >= 0 && roundedX_blue < paperWidth && roundedY_blue >= 0 && roundedY_blue < paperHeight)
        {
            collisionCount++;
            paper[roundedY_blue][roundedX_blue][0] = (1.0f - blendCoefficient) * paper[roundedY_blue][roundedX_blue][0] + blendCoefficient * 0.0f;
            paper[roundedY_blue][roundedX_blue][1] = (1.0f - blendCoefficient) * paper[roundedY_blue][roundedX_blue][1] + blendCoefficient * 0.0f;
            paper[roundedY_blue][roundedX_blue][2] = (1.0f - blendCoefficient) * paper[roundedY_blue][roundedX_blue][2] + blendCoefficient * 1.0f;
        }
    }

    // Check Elapsed Time
	auto sprayEnd = std::chrono::high_resolution_clock::now();
	elapsed = sprayEnd - sprayConeEnd;
	std::cout << "Time Taken to Check for Collisions Between the Paper and Particles, and to Blend the Colours: " << elapsed.count() << "s\n";

    bitmap_image image(paperWidth, paperHeight);

    for (int i = 0; i < paperHeight; i++) {
        for (int j = 0; j < paperWidth; j++) {
            unsigned char red = static_cast<unsigned char>(paper[i][j][0] * 255);
            unsigned char green = static_cast<unsigned char>(paper[i][j][1] * 255);
            unsigned char blue = static_cast<unsigned char>(paper[i][j][2] * 255);

            image.set_pixel(j, i, red, green, blue);
        }
    }
    image.save_image("spray_image.bmp");

    // Check Elapsed Time
    auto imageSaveEnd = std::chrono::high_resolution_clock::now();
    elapsed = imageSaveEnd - sprayEnd;
    std::cout << "Time taken to Save The Produced Image: " << elapsed.count() << "s\n";

    // Print Simulation Statistics
	std::wcout << "Number of Particles Created: " << numOfParticles * 3 << std::endl;
	std::wcout << "Number of Particles that Collided with the Paper: " << collisionCount << std::endl;
	std::wcout << "Number of Particles that didn't Collide with the Paper: " << numOfParticles * 3 - collisionCount << std::endl;

    // Deallocate Memory
    for (int i = 0; i < paperHeight; ++i) {
        for (int j = 0; j < paperWidth; ++j) {
            delete[] paper[i][j];
        }
        delete[] paper[i];
    }
    delete[] paper;

    delete[] spray_red;
    delete[] spray_green;
    delete[] spray_blue;

    return 0;
}
