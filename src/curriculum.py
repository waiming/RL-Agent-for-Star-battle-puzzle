from star_battle_env import StarBattleEnv
from region_presets import custom_regions_6x6

def curriculum_env(level):
    if level == 0:
        return StarBattleEnv(grid_size=4, stars_per_row=1)
    elif level == 1:
        return StarBattleEnv(grid_size=6, stars_per_row=2, regions=custom_regions_6x6)
    elif level == 2:
        return StarBattleEnv(grid_size=8, stars_per_row=2)  # Add real region presets later
    else:
        return StarBattleEnv(grid_size=10, stars_per_row=2)  # Add real region presets later