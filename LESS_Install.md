### 1. System Requirements
    Hardware:
        Minimum 16GB RAM
        Dedicated GPU (required for 3D rendering)
    Software:
        UE4 version ≥ 4.25 (4.25 and above are recommended for compatibility)
        Epic Games account

### 2. UE4 Installation Steps
#### 2.1 Clone UE4 Source Code
```bash
# Navigate to your GitHub projects folder
git clone -b 4.27 git@github.com:EpicGames/UnrealEngine.git
cd UnrealEngine
```
#### 2.2 Build UE4
```bash
./Setup.sh
./GenerateProjectFiles.sh
make
```

Note: Ensure you've registered an Epic Games account and applied for UE4 source code access.
### 3. AirSim Installation Steps
#### 3.1 Clone AirSim Repository
```bash

# Navigate to your GitHub projects folder
git clone https://github.com/Microsoft/AirSim.git
cd AirSim
```
#### 3.2 Build AirSim
```bash
./setup.sh
./build.sh
```


For detailed installation guidance, refer to: https://zhuanlan.zhihu.com/p/449619917

#### 3.3 Post-Installation Setup

Download the LESS project to your computer.
    Replace ./Documents/AirSim/settings.json with the file from the installation package(LESS_Json).
    Open the LESS project (expect longer load times due to project size).

### 4. Controlling the Lunar Rover

After opening the LESS project, click **Play**.
Use keyboard arrow keys (↑↓←→) to control the lunar rover movement.
