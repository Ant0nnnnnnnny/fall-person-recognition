import argparse
def parse_args():

    # ===============================Project ============================
    parser = argparse.ArgumentParser(description='Arguments for person pose estimation.')
    parser.add_argument('--seed',type = int, default=7310,help = 'Random seed.')
    

    # ===============================Data ================================



    return parser
