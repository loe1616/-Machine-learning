"""The template of the main script of the machine learning process
"""

import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():
    """The main loop of the machine learning process

    This loop is run in a separate process, and communicates with the game process.

    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here.

    # 2. Inform the game process that ml process is ready before start the loop.
    # import pickle
    # import numpy as np
    # filename = "C:\\Users\\loe_lin\\Documents\\-Machine-learning\\MLGame-master\\games\\arkanoid\\ml\\knn_example.sav"
    # model = pickle.load(open(filename,'rb'))
    comm.ml_ready()
    
    
    
    ball_position_history = []

    # 3. Start an endless loop.
    while True:
        # 3.1. Receive the scene information sent from the game process.
        scene_info = comm.get_scene_info()
        ball_position_history.append(scene_info.ball)
        Platform_center_x = scene_info.platform[0]+20
        if(len(ball_position_history))==1:
            ball_going_down = 0
        elif ball_position_history[-1][1]-ball_position_history[-2][1] > 0:
            ball_going_down = 1
            vy = ball_position_history[-1][1]-ball_position_history[-2][1]
            vx = ball_position_history[-1][0]-ball_position_history[-2][0]
            ball_destination = ball_position_history[-1][0]+(395-ball_position_history[-1][1])*vx/vy

            while ball_destination < 0 or ball_destination > 200:
                if ball_destination >200:
                    ball_destination = 400 - ball_destination
                else:
                    ball_destination =  - ball_destination

            if Platform_center_x < ball_destination:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT) 
        else:
            bell_going_down = 0 
            if Platform_center_x < 100:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
            else:
                comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)                          

        # 3.2. If the game is over or passed, the game process will reset
        #      the scene and wait for ml process doing resetting job.
        # Platform_center_x = scene_info.platform[0]+20
        # inp_temp = np.array([scene_info.ball[0],scene_info.ball[1],scene_info.platform[0]])
        # input = inp_temp[np.newaxis,:]


        if scene_info.status == GameStatus.GAME_OVER or \
            scene_info.status == GameStatus.GAME_PASS:
            # Do some stuff if needed
            #scene_info = comm.get_scene_infso()
            # 3.2.1. Inform the game process that ml process is ready
            comm.ml_ready()
            continue
        
        # move = model.predict(input)
        # if move < 0:
        #     comm.send_instruction(scene_info.frame,PlatformAction.MOVE_LEFT)
        # elif move > 0:
        #     comm.send_instruction(scene_info.frame,PlatformAction.MOVE_RIGHT)
        # else :
        #     comm.send_instruction(scene_info.frame,PlatformAction.NONE)

        # 3.3. Put the code here to handle the scene information

        # 3.4. Send the instruction for this frame to the game process
          