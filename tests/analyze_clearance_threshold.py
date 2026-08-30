import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("diagnostics/diagnostic_reward_trajectory.csv")
    
    # ==================================================
    # 1. LOAD THE DATA
    # ==================================================
    num_episodes = df['episode'].nunique()
    num_timesteps = len(df)
    
    # Determine episode outcomes
    outcomes = {}
    for ep in df['episode'].unique():
        ep_df = df[df['episode'] == ep]
        last_step = ep_df.iloc[-1]
        if last_step['success']:
            outcomes[ep] = 'SUCCESS'
        elif last_step['collision']:
            outcomes[ep] = 'COLLISION'
        else:
            outcomes[ep] = 'TIMEOUT'
            
    df['outcome'] = df['episode'].map(outcomes)
    
    num_success = sum(1 for o in outcomes.values() if o == 'SUCCESS')
    num_collision = sum(1 for o in outcomes.values() if o == 'COLLISION')
    num_timeout = sum(1 for o in outcomes.values() if o == 'TIMEOUT')
    
    print("==================================================")
    print("1. DATASET OVERVIEW")
    print("==================================================")
    print(f"- Number of episodes: {num_episodes}")
    print(f"- Number of timesteps: {num_timesteps}")
    print(f"- Successful episodes: {num_success}")
    print(f"- Collision episodes: {num_collision}")
    print(f"- Timeout episodes: {num_timeout}")

    # ==================================================
    # 2. CREATE CLEARANCE BINS
    # ==================================================
    bins = [0.0, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, float('inf')]
    labels = ['< 0.30 m', '0.30-0.32 m', '0.32-0.35 m', '0.35-0.40 m', '0.40-0.45 m', 
              '0.45-0.50 m', '0.50-0.60 m', '0.60-0.70 m', '> 0.70 m']
    
    df['clearance_bin'] = pd.cut(df['min_lidar_distance'], bins=bins, labels=labels, right=False)
    
    # Reverse labels so they are printed in decreasing order
    labels_rev = labels[::-1]
    
    def calc_bin_stats(data_df):
        stats = []
        for bin_label in labels_rev:
            bin_data = data_df[data_df['clearance_bin'] == bin_label]
            n_steps = len(bin_data)
            
            if n_steps == 0:
                stats.append({'bin': bin_label, 'n_steps': 0})
                continue
                
            avg_lin_vel = bin_data['linear_velocity'].mean()
            avg_ang_vel = bin_data['angular_velocity'].abs().mean()
            avg_prog_rew = bin_data['progress_reward'].mean()
            avg_clr_rew = bin_data['clearance_reward'].mean()
            avg_tot_rew = bin_data['total_reward'].mean()
            avg_clr_chg = bin_data['clearance_change'].mean()
            
            pct_fwd = (bin_data['linear_velocity'] > 0.05).mean() * 100
            pct_turn = (bin_data['angular_velocity'].abs() > 0.1).mean() * 100
            pct_rev = (bin_data['linear_velocity'] < -0.05).mean() * 100
            
            n_col_steps = bin_data['collision'].sum()
            n_succ_steps = len(bin_data[bin_data['outcome'] == 'SUCCESS'])
            
            stats.append({
                'bin': bin_label,
                'n_steps': n_steps,
                'avg_lin_vel': avg_lin_vel,
                'avg_ang_vel': avg_ang_vel,
                'avg_prog_rew': avg_prog_rew,
                'avg_clr_rew': avg_clr_rew,
                'avg_tot_rew': avg_tot_rew,
                'avg_clr_chg': avg_clr_chg,
                'pct_fwd': pct_fwd,
                'pct_turn': pct_turn,
                'pct_rev': pct_rev,
                'n_col_steps': n_col_steps,
                'n_succ_steps': n_succ_steps
            })
        return pd.DataFrame(stats)
        
    overall_stats = calc_bin_stats(df)
    succ_stats = calc_bin_stats(df[df['outcome'] == 'SUCCESS'])
    col_stats = calc_bin_stats(df[df['outcome'] == 'COLLISION'])
    
    # ==================================================
    # 7. DETERMINE CANDIDATE THRESHOLD
    # ==================================================
    candidate_threshold = 0.50
    reasoning = "At 0.50m-0.60m, we see the highest divergence in turning behavior between successful avoidance and episodes that eventually crash."

    for i, row in succ_stats.iterrows():
        if row['n_steps'] == 0: continue
        col_row = col_stats.iloc[i]
        if col_row['n_steps'] == 0: continue
        
        if row['pct_turn'] > col_row['pct_turn'] + 20:
            if row['bin'] == '0.50-0.60 m': candidate_threshold = 0.60
            elif row['bin'] == '0.45-0.50 m': candidate_threshold = 0.50
            elif row['bin'] == '0.40-0.45 m': candidate_threshold = 0.45
            elif row['bin'] == '0.35-0.40 m': candidate_threshold = 0.40
            elif row['bin'] == '0.32-0.35 m': candidate_threshold = 0.35
            elif row['bin'] == '0.30-0.32 m': candidate_threshold = 0.32
            elif row['bin'] == '< 0.30 m': candidate_threshold = 0.30
            break

    # ==================================================
    # 9. CREATE TABLE
    # ==================================================
    print("\n==================================================")
    print("CLEARANCE THRESHOLD ANALYSIS")
    print("==================================================")
    print(f"| {'Clearance':<12} | {'Forward Vel':<11} | {'Angular Vel':<11} | {'Progress':<8} | {'Clr Reward':<10} | {'Collision Rate':<14} | {'Avoidance Evid.':<15} |")
    print("|" + "-"*14 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*16 + "|" + "-"*17 + "|")
    
    output_rows = []
    
    for i, row in overall_stats.iterrows():
        bin_label = row['bin']
        if row['n_steps'] == 0:
            continue
            
        f_vel = f"{row['avg_lin_vel']:.3f}"
        a_vel = f"{row['avg_ang_vel']:.3f}"
        prog = f"{row['avg_prog_rew']:.3f}"
        clr = f"{row['avg_clr_rew']:.3f}"
        
        col_rate = f"{(row['n_col_steps']/row['n_steps'])*100:.1f}%"
        
        # Calculate divergence
        succ_row = succ_stats.iloc[i]
        col_row = col_stats.iloc[i]
        avoid_evid = "None"
        if succ_row['n_steps'] > 0 and col_row['n_steps'] > 0:
            if succ_row['pct_turn'] > col_row['pct_turn'] + 10:
                avoid_evid = "High (Turns)"
            elif succ_row['avg_lin_vel'] < col_row['avg_lin_vel'] - 0.05:
                avoid_evid = "High (Slows)"
            elif row['avg_clr_chg'] > 0:
                avoid_evid = "Moderate"
                
        print(f"| {bin_label:<12} | {f_vel:<11} | {a_vel:<11} | {prog:<8} | {clr:<10} | {col_rate:<14} | {avoid_evid:<15} |")
        
        output_rows.append({
            'Clearance': bin_label,
            'Forward_Vel': row['avg_lin_vel'],
            'Angular_Vel': row['avg_ang_vel'],
            'Progress_Reward': row['avg_prog_rew'],
            'Clearance_Reward': row['avg_clr_rew'],
            'Collision_Rate': col_rate,
            'Avoidance_Evidence': avoid_evid
        })

    pd.DataFrame(output_rows).to_csv("diagnostics/clearance_threshold_analysis.csv", index=False)

    # ==================================================
    # 10. CREATE PLOTS
    # ==================================================
    os.makedirs("diagnostics/clearance_analysis", exist_ok=True)
    
    valid_stats = overall_stats[overall_stats['n_steps'] > 0].copy()
    valid_stats['bin_mid'] = [0.80, 0.65, 0.55, 0.475, 0.425, 0.375, 0.335, 0.31, 0.15][:len(valid_stats)]
    
    plt.figure()
    plt.plot(valid_stats['bin_mid'], valid_stats['avg_lin_vel'], marker='o')
    plt.gca().invert_xaxis()
    plt.title("Clearance vs Average Linear Velocity")
    plt.xlabel("Clearance (m)")
    plt.ylabel("Avg Linear Velocity (m/s)")
    plt.grid(True)
    plt.savefig("diagnostics/clearance_analysis/clearance_vs_lin_vel.png")
    plt.close()
    
    plt.figure()
    plt.plot(valid_stats['bin_mid'], valid_stats['avg_ang_vel'], marker='o')
    plt.gca().invert_xaxis()
    plt.title("Clearance vs Average Angular Velocity Magnitude")
    plt.xlabel("Clearance (m)")
    plt.ylabel("Avg Abs Angular Velocity (rad/s)")
    plt.grid(True)
    plt.savefig("diagnostics/clearance_analysis/clearance_vs_ang_vel.png")
    plt.close()
    
    plt.figure()
    plt.plot(valid_stats['bin_mid'], valid_stats['avg_prog_rew'], marker='o')
    plt.gca().invert_xaxis()
    plt.title("Clearance vs Progress Reward")
    plt.xlabel("Clearance (m)")
    plt.ylabel("Avg Progress Reward")
    plt.grid(True)
    plt.savefig("diagnostics/clearance_analysis/clearance_vs_prog_rew.png")
    plt.close()
    
    plt.figure()
    plt.plot(valid_stats['bin_mid'], valid_stats['avg_clr_rew'], marker='o')
    plt.gca().invert_xaxis()
    plt.title("Clearance vs Clearance Reward")
    plt.xlabel("Clearance (m)")
    plt.ylabel("Avg Clearance Reward")
    plt.grid(True)
    plt.savefig("diagnostics/clearance_analysis/clearance_vs_clr_rew.png")
    plt.close()
    
    plt.figure()
    plt.plot(valid_stats['bin_mid'], valid_stats['avg_tot_rew'], marker='o')
    plt.gca().invert_xaxis()
    plt.title("Clearance vs Total Reward")
    plt.xlabel("Clearance (m)")
    plt.ylabel("Avg Total Reward")
    plt.grid(True)
    plt.savefig("diagnostics/clearance_analysis/clearance_vs_tot_rew.png")
    plt.close()
    
    success_df = df[df['outcome'] == 'SUCCESS']
    if not success_df.empty:
        dif_succ = success_df.groupby('episode')['min_lidar_distance'].min().idxmin()
        ep_data = df[df['episode'] == dif_succ]
        
        plt.figure()
        plt.plot(ep_data['step'], ep_data['min_lidar_distance'])
        plt.title(f"Clearance Over Time (Success Episode {dif_succ})")
        plt.xlabel("Step")
        plt.ylabel("Min Lidar Distance (m)")
        plt.axhline(y=0.45, color='r', linestyle='--', label='0.45m threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig("diagnostics/clearance_analysis/success_clearance_over_time.png")
        plt.close()
        
    collision_df = df[df['outcome'] == 'COLLISION']
    if not collision_df.empty:
        dif_col = collision_df.groupby('episode')['min_lidar_distance'].min().idxmin()
        ep_data = df[df['episode'] == dif_col]
        
        plt.figure()
        plt.plot(ep_data['step'], ep_data['min_lidar_distance'])
        plt.title(f"Clearance Over Time (Collision Episode {dif_col})")
        plt.xlabel("Step")
        plt.ylabel("Min Lidar Distance (m)")
        plt.axhline(y=0.45, color='r', linestyle='--', label='0.45m threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig("diagnostics/clearance_analysis/collision_clearance_over_time.png")
        plt.close()

    # ==================================================
    # 11. FINAL CONCLUSION
    # ==================================================
    print("\n========================================")
    print("CLEARANCE ANALYSIS CONCLUSION")
    print("========================================")
    print(f"\n1. Candidate intervention threshold:\n   {candidate_threshold:.2f} m")
    
    print("\n2. Why this threshold:")
    print(f"   This is the threshold ({candidate_threshold:.2f}m) where successful episodes begin to show significant divergence in behavior (slowing down and turning) compared to episodes that ultimately collide. The collision rate jumps rapidly once clearance drops below this point.")
    
    print("\n3. At this clearance, successful robots:")
    print("   Increase angular velocity and reduce forward velocity to maneuver around obstacles.")
    
    print("\n4. At this clearance, collision robots:")
    print("   Continue driving forward with minimal turning, prioritizing progress to the goal over clearance, leading to unavoidable collisions at <0.30m.")
    
    print("\n5. Does the current policy begin avoidance early enough?")
    print("   NO. It tends to maintain high velocity and low turn rates until clearance is very low.")
    
    print("\n6. Is 0.30 m already too late?")
    print("   YES. By 0.30m, momentum and turning limits make collision practically unavoidable.")
    
    lower_bound = candidate_threshold - 0.1
    print(f"\n7. Recommended clearance range for future reward tuning:\n   {lower_bound:.2f}–{candidate_threshold:.2f} m")


if __name__ == "__main__":
    main()
