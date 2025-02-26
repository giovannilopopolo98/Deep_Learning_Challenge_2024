import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)   
    parser.add_argument("--max_epochs", type=int, default=100)
    
    parser.add_argument("--ckpt_dir", type=str, default="D:/UNIVERSITA/MAGISTRALE/SECONDO_ANNO/PRIMO_SEMESTRE/Deep_Learning/Script/CHALLENGE/CHALLENGE/checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="depth")
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join("D:\\", "UNIVERSITA", "MAGISTRALE", "SECONDO_ANNO", "PRIMO_SEMESTRE", "Deep_Learning", "Script", "CHALLENGE", "DepthEstimationUnreal"))

    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--ckpt_file", type=str, default="D:/UNIVERSITA/MAGISTRALE/SECONDO_ANNO/PRIMO_SEMESTRE/Deep_Learning/Script/CHALLENGE/CHALLENGE/checkpoint/depth_v_0.4.1_196.pth")

    args = parser.parse_args()
    solver = Solver(args)
    if args.is_train:
        solver.fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
