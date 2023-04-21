
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def r2euler(R, type):
    R = np.array(R)
    type = str(type).upper()
    err = float(0.001)

    if type == "XYZ":
        # R[0,2]/sqrt((R[1,2])**2 + (R[2,2])**2) == sin(beta)/|cos(beta)| 
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(R[0,2], math.sqrt((R[1,2])**2 + (R[2,2])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,0], R[1,1])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,0], R[1,1])
        else:
            alpha = math.atan2(-(R[1,2])/(math.cos(beta)),(R[2,2])/(math.cos(beta)))
            gamma = math.atan2(-(R[0,1])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))

    elif type == "XZY":
        # -R[0,1]/sqrt((R[1,1])**2 + (R[2,1])**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(-R[0,1], math.sqrt((R[1,1])**2 + (R[2,1])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,2], R[1,0])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], -R[1,0])
        else:
            alpha = math.atan2((R[2,1])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))
            gamma = math.atan2((R[0,2])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))

    elif type == "YXZ":
        # -R[1,2]/sqrt(R[0,2]**2 + R[2,2]**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(-R[1,2], math.sqrt((R[0,2])**2 + (R[2,2])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        else:
            alpha = math.atan2((R[0,2])/(math.cos(beta)), (R[2,2])/(math.cos(beta)))
            gamma = math.atan2((R[1,0])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))

    elif type == "YZX":
        # R[1,0]/sqrt(R[0,0]**2 + R[2,0]**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(R[1,0], math.sqrt((R[0,0])**2 + (R[2,0])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], -R[0,1])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,2], R[0,1])
        else:
            alpha = math.atan2(-(R[2,0])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))
            gamma = math.atan2(-(R[1,2])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))

    elif type == "ZXY":
        # R[2,1]/sqrt(R[0,1]**2 + R[1,1]**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(R[2,1], math.sqrt((R[0,1])**2 + (R[1,1])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        else:
            alpha = math.atan2(-(R[0,1])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))
            gamma = math.atan2(-(R[2,0])/(math.cos(beta)), (R[2,2])/(math.cos(beta)))

    elif type == "ZYX":
        # -R[2,0]/sqrt(R[0,0]**2 + R[1,0]**2) == sin(beta)/|cos(beta)| 
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(-R[2,0], math.sqrt((R[0,0])**2 + (R[1,0])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,1], R[1,2])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], -R[1,2])
        else:
            alpha = math.atan2((R[1,0])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))
            gamma = math.atan2((R[2,1])/(math.cos(beta)), (R[2,2])/(math.cos(beta)))
    
    elif type == "XYX":
        # sqrt(R[0,1]**2 + R[0,2]**2)/R[0,0] == |sin(beta)|/cos(beta)
        # ==> beta (0, pi)
        beta = math.atan2(math.sqrt((R[0,1])**2 + (R[0,2])**2), R[0,0])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], R[1,1])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], R[1,1])
        else:
            alpha = math.atan2((R[1,0])/(math.sin(beta)), -(R[2,0])/(math.sin(beta)))
            gamma = math.atan2((R[0,1])/(math.sin(beta)), (R[0,2])/(math.sin(beta)))

    elif type == "XZX":
        # sqrt(R[1,0]**2 + R[2,0]**2)/R[0,0] == |sin(beta)|/cos(beta)
        # ==> beta (0, pi)
        beta = math.atan2(math.sqrt((R[1,0])**2 + (R[2,0])**2), R[0,0])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], R[1,1])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,2], -R[1,1])
        else:
            alpha = math.atan2((R[2,0])/(math.sin(beta)), (R[1,0])/(math.sin(beta)))
            gamma = math.atan2((R[0,2])/(math.sin(beta)), -(R[0,1])/(math.sin(beta)))

    elif type == "YXY":
        # sqrt(R[0,1]**2 + R[2,1]**2)/R[1,1] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,1])**2 + (R[2,1])**2), R[1,1])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        else:
            alpha = math.atan2((R[0,1])/(math.sin(beta)), (R[2,1])/(math.sin(beta)))
            gamma = math.atan2((R[1,0])/(math.sin(beta)), -(R[1,2])/(math.sin(beta)))

    elif type == "YZY":
        # sqrt(R[0,1]**2 + R[2,1]**2)/R[1,1] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,1])**2 + (R[2,1])**2), R[1,1])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,2], -R[0,0])
        else:
            alpha = math.atan2((R[2,1])/(math.sin(beta)), -(R[0,1])/(math.sin(beta)))
            gamma = math.atan2((R[1,2])/(math.sin(beta)), (R[1,0])/(math.sin(beta)))

    elif type == "ZXZ":
        # sqrt(R[0,2]**2 + R[1,2]**2)/R[2,2] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,2])**2 + (R[1,2])**2), R[2,2])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        else:
            alpha = math.atan2((R[0,2])/(math.sin(beta)), -(R[1,2])/(math.sin(beta)))
            gamma = math.atan2((R[2,0])/(math.sin(beta)), (R[2,1])/(math.sin(beta)))

    elif type == "ZYZ":
        # sqrt(R[0,2]**2 + R[1,2]**2)/R[2,2] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,2])**2 + (R[1,2])**2), R[2,2])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        elif beta >= math.pi+err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,1], -R[0,0])
        else:
            alpha = math.atan2((R[1,2])/(math.sin(beta)), (R[0,2])/(math.sin(beta)))
            gamma = math.atan2((R[2,1])/(math.sin(beta)), -(R[2,0])/(math.sin(beta)))
    
    return alpha, beta, gamma


if __name__ == "__main__":
    RM = np.loadtxt ('RotationMatrix.txt')
    print("Upper -- intrinsic\nLower -- extrinsic")

    euler_type = str(input("Please input the type of euler angle...\n"))

    if euler_type.isupper():
        angle_0, angle_1, angle_2 = r2euler(RM, euler_type)
        print("Intrinsic")
        print("Angle about {} is {}".format(euler_type[0], angle_0))
        print("Angle about {} is {}".format(euler_type[1], angle_1))
        print("Angle about {} is {}".format(euler_type[2], angle_2))
    elif euler_type.islower():
        angle_0, angle_1, angle_2 = r2euler(RM, euler_type.upper()[::-1])[::-1]
        print("extrinsic")
        print("Angle about {} is {}".format(euler_type[0], angle_0))
        print("Angle about {} is {}".format(euler_type[1], angle_1))
        print("Angle about {} is {}".format(euler_type[2], angle_2))
    else:
        pass
