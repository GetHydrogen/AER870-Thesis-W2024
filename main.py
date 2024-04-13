from utils import *



METHOD = "FBDS" #{"LKSP", "FBDS"}
FRAMERATE = 30
DEPTH_CORRECTION = 2.0

def main():

    if METHOD == "LKSP":
        velocities_data = LucasKanade_Sparse(
            fps = FRAMERATE, 
            Z   = DEPTH_CORRECTION,
            )
    elif METHOD == "FBDS":
        velocities_data = Farneback_Dense(
            fps = FRAMERATE, 
            Z   = DEPTH_CORRECTION,
        )

    # Log Files
    logFiles(
        velocities_data=velocities_data,
        method = METHOD,
        trial_id = TRIAL_ID,
    )

    
    return

if __name__ == "__main__":
    main()
