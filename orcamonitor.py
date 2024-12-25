#!/home/jingdan/apps/miniconda3/bin/python
import re
import sys, os
import numpy as np
import argparse

# Regular expression for matching the:
# Redundant Internal Coordinates (Angstroem and degrees)
# Definition  Value    dE/dq     Step     New-Value  [comp.(TS mode)]
pattern = r'^\s*(\d+)\.\s+([BADL]\(([A-Za-z]+\s+\d+,?)+(?:\s+\d+)?\))\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?'

# extract the atom and the number: 'B(C   1,C   0)' --> [('C', '1'), ('C', '0')]
pattern2 = r'([A-Za-z])+\s+(\d+),?'

# Regular expression for matching the optimization cycle beginning
pattern3 = r'^\s*\*\s*GEOMETRY OPTIMIZATION CYCLE\s*(\d+)\s*\*'

# Regular expression for matching the imaginary mode originated from Hessian Calculation
pattern4 = r'^\s+(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1\s+(\*\*\*imaginary mode\*\*\*)?'

# Regular expression for matching charge and multiplicity
pattern5 = r'^\|\s+\d+>\s+\*\s*(?:XYZ|XYZFILE)\s+(\d+)\s+(\d+)'

# Regular expression for matching the optimization convergence
pattern6_dict = {
    "E_change": r'^\s+Energy\schange\s+(-?\d+\.\d+)\s+(?:-?\d+\.\d+)\s+(?:YES|NO)',
    "RMS_grad": r'^\s+RMS\sgradient\s+(-?\d+\.\d+)\s+(?:-?\d+\.\d+)\s+(?:YES|NO)',
    "MAX_grad": r'^\s+MAX gradient\s+(-?\d+\.\d+)\s+(?:-?\d+\.\d+)\s+(?:YES|NO)',
    "RMS_step": r'^\s+RMS\sstep\s+(-?\d+\.\d+)\s+(?:-?\d+\.\d+)\s+(?:YES|NO)',
    "MAX_step": r'^\s+MAX\sstep\s+(-?\d+\.\d+)\s+(?:-?\d+\.\d+)\s+(?:YES|NO)',
}

# Regular expression for matching the num of atoms
# Number of atoms                             ...     26
pattern7 = r'^Number\sof\satoms\s+\.\.\.\s+(\d+)'

# Extract minimum eigen value of the Hessian matrix
pattern8 = r'Lowest eigenvalues of (?:augmented|the) Hessian:\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)\s+(-?\d*\.\d+)'

opt_flag_txt = "GEOMETRY OPTIMIZATION CYCLE"
ts_flag_txt = "TS mode is"
freq_flag_txt = "THERMOCHEMISTRY AT"




def get_read_internal_coords(lines, ts_flag=True):
    
    all_matches = []
    for line_num, line in enumerate(lines, start=1):
        # Search for matches in the current line
        match = re.search(pattern, line)
        if match:
            # If a match is found, print the match and the line number
            match_lis = list(match.groups())
            
            all_matches.append([line_num, match_lis])
    
    def progress_match_item(match_item):
        internal_item = match_item[1] # string
        assert internal_item[0] == 'B' or internal_item[0] == 'A' or internal_item[0] == 'D' or internal_item[0] == 'L', "Internal coordinate type not recognized"
        
        temp_lis = re.findall(pattern2, internal_item)
        temp_lis = [internal_item[0]] + [item[1] for item in temp_lis]
        new_internal_item = "-".join(temp_lis)
        
        res = [int(match_item[0]), new_internal_item, float(match_item[3]), float(match_item[4]), float(match_item[5]), float(match_item[6])]
        
        if ts_flag:
            if match_item[7]:
                res.append(float(match_item[7]))
            else:
                res.append(0.)
        return res
    
    all_matches = list(map(lambda a:[a[0], progress_match_item(a[1])], 
                           all_matches))
    all_matches_line_idx = [item[0] for item in all_matches]
    all_matches_cont = [item[1] for item in all_matches]
    
    internal_idx_lis = [item[0] for item in all_matches_cont]
    internal_coordName_lis = [item[1] for item in all_matches_cont]
    
    internal_size = max(internal_idx_lis)
    internal_coord_list = internal_coordName_lis[:internal_size]
    
    number_mat_lis = [item[2:] for item in all_matches_cont]
    number_mat_arr = np.array(number_mat_lis, dtype=float)
    
    mat_columns = ['Value', 'dE/dq', 'Step', 'New-Value'] if not ts_flag else ['Value', 'dE/dq', 'Step', 'New-Value', 'comp.(TS mode)']
    number_mat_arr = number_mat_arr.reshape(-1, internal_size, len(mat_columns))
    
    
    line_idx_begin_end = [(all_matches_line_idx[i], all_matches_line_idx[i+internal_size-1]) for i in range(0, len(all_matches_line_idx), internal_size)]
    
    # output: internal_coord_list, mat_columns, line_idx_begin_end, number_mat_arr
    # internal_coord_list: list of internal coord names
    # mat_columns: list of column names, for ts, 5, for int 4
    # line_idx_begin_end: list of tuples, each tuple is the begin and end line number of a set of internal coordinates
    # number_mat_arr: 3D array, (n, m, k), n is the number of internal coord sets, m is the number of internal coords, k is the number of columns
    return internal_coord_list, mat_columns, line_idx_begin_end, number_mat_arr


def get_ts_mode(lines, n_modes=2):
    full_text = ''.join(lines)
    
    if opt_flag_txt in full_text:
        if ts_flag_txt in full_text:
            ts_flag = True
        else:
            ts_flag = False
    else:
        raise ValueError("Not Optimization Output")
    
    internal_coord_list, mat_columns, line_idx_begin_end, number_mat_arr = \
        get_read_internal_coords(lines, ts_flag=ts_flag)
    assert number_mat_arr.shape == (len(line_idx_begin_end), len(internal_coord_list), len(mat_columns)), "Shape mismatch"
    
    ts_mode_col = number_mat_arr[:, :, -1]
    
    max_n_ts_mode_idx = np.argsort(ts_mode_col, axis=1)[:, -n_modes:][:, ::-1]
    
    max_ts_mode = [", ".join([internal_coord_list[i] for i in sublis]) for sublis in max_n_ts_mode_idx  ]
    
    result = [[line_idx_begin_end[i][0], max_ts_mode[i], "redundant_internal"] for i in range(len(line_idx_begin_end))]
    return result

def get_opt_cycle_begin(lines):
    all_matches = []
    for line_num, line in enumerate(lines, start=1):
        # Search for matches in the current line
        match = re.search(pattern3, line)
        if match:
            all_matches.append([line_num, int(match.groups()[0]), "opt_cycle_begin"])
            
    return all_matches

def get_cartesian_coords_begin(lines):
    all_matches = []
    for line_num, line in enumerate(lines, start=1):
        # Search for matches in the current line
        match = re.search(r'^CARTESIAN COORDINATES \(ANGSTROEM\)', line)
        if match:
            all_matches.append(line_num + 1)
            
    return all_matches
    
def get_imaginary_freq(lines):
    
    all_matches = []
    for line_num, line in enumerate(lines, start=1):
        # Search for matches in the current line
        match = re.search(pattern4, line)
        if match:
            match_lis = list(match.groups())
            all_matches.append([line_num, match_lis])
            
    
    def progress_match_item(match_item):
        frequency = float(match_item[1])
        if match_item[2]:
            # flag = 1 if match_item[2] == "***imaginary mode***" else -1
            if match_item[2] == "***imaginary mode***":
                flag = True
            else:
                raise ValueError("Unknown imaginary mode information:{match_item[2]}")
            # 1 for imaginary mode, -1 for wrong information?
        else:
            flag = False # 0 for normal mode
            
        res = [int(match_item[0]), frequency, flag]
        return res
    
    all_matches = list(map(lambda a:[a[0], progress_match_item(a[1])], 
                           all_matches))
    all_matches_line_idx = [item[0] for item in all_matches]
    all_matches_cont = [item[1] for item in all_matches]
    
    mode_idx_lis = [item[0] for item in all_matches_cont]
    
    mode_size = max(mode_idx_lis) + 1
    
    all_matches_freq_arr = np.array([item[1] for item in all_matches_cont], dtype=float)
    all_matches_freq_arr = all_matches_freq_arr.reshape(-1, mode_size)
    
    all_matches_flag_arr = np.array([item[2] for item in all_matches_cont], dtype=bool)
    all_matches_flag_arr = all_matches_flag_arr.reshape(-1, mode_size)
    
    line_idx_begin_end = [(all_matches_line_idx[i], all_matches_line_idx[i+mode_size-1]) for i in range(0, len(all_matches_line_idx), mode_size)]
    return line_idx_begin_end, all_matches_freq_arr, all_matches_flag_arr

def get_imaginary_mode(lines):
    line_idx_begin_end, all_matches_freq_arr, all_matches_flag_arr = \
        get_imaginary_freq(lines)
    
    imaginary_freq = [all_matches_freq_arr[i][all_matches_flag_arr[i]] for i in range(all_matches_freq_arr.shape[0])]
    imaginary_freq = list(map(str, imaginary_freq))
    
    result = [[line_idx_begin_end[i][0], imaginary_freq[i], "imaginary_mode"] for i in range(len(line_idx_begin_end))]
    return result

def get_opt_convergence(lines, get_ratio=True):
    pattern6_match_res = {
        key: [] for key in pattern6_dict
    }
    for line_num, line in enumerate(lines, start=1):
        # Search for matches in the current line
        match = re.search(pattern6_dict["RMS_grad"], line)
        if match:
            pattern6_match_res["RMS_grad"].append(float(match.groups()[0]))
            match_E = re.search(pattern6_dict["E_change"], lines[line_num-2])
            pattern6_match_res["E_change"].append(float(match_E.groups()[0]) if match_E else np.nan)
            match_MAX_grad = re.search(pattern6_dict["MAX_grad"], lines[line_num])
            pattern6_match_res["MAX_grad"].append(float(match_MAX_grad.groups()[0]))
            match_RMS_step = re.search(pattern6_dict["RMS_step"], lines[line_num+1])
            pattern6_match_res["RMS_step"].append(float(match_RMS_step.groups()[0]))
            match_MAX_step = re.search(pattern6_dict["MAX_step"], lines[line_num+2])
            pattern6_match_res["MAX_step"].append(float(match_MAX_step.groups()[0]))
    

    opt_conver_normal = {
        "E_change": 0.0000050000,
        "RMS_grad": 0.0001000000,
        "MAX_grad": 0.0003000000,
        "RMS_step": 0.0020000000,
        "MAX_step": 0.0040000000,
    }

    opt_conver_ratio = {
        key: np.array(pattern6_match_res[key]) / value
            for key, value in opt_conver_normal.items()
    }
    
    if get_ratio:
        return opt_conver_ratio, opt_conver_ratio
    else:
        return pattern6_match_res, opt_conver_ratio


def output_pop(lines, output_filename='', output_ratio=True, format_out = [3, 6, 6, 6, 6, 6, 22, 24, 17, 5]):
    opt_cycle_begin = get_opt_cycle_begin(lines)
    opt_conver, opt_conver_rat = get_opt_convergence(lines, get_ratio=output_ratio)
    min_eigen = get_hessian_min_eigen(lines)
    job_type = get_task_type(lines)
    del_col = []
    
    if job_type != "TS":
        del_col.append(6) # delete the "im_mode"
        ts_mode = []
    else:
        ts_mode = get_ts_mode(lines) # not this if it's not ts job
    
    if freq_flag_txt in "".join(lines):
        imaginary_mode = get_imaginary_mode(lines) # not this if we do not have frequency calculation during optimization
    else:
        del_col.append(7) # delete the "im_freq"
        imaginary_mode = []

    def get_res_cache(res_cache):
        temp = res_cache[1:]
        if 7 in del_col:
            temp.pop(1)
        if 6 in del_col:
            temp.pop(0)
        return temp

    
    temp_list = opt_cycle_begin + imaginary_mode + ts_mode
    temp_list = sorted(temp_list, key=lambda x: x[0])[::-1]
    
    
    result = [["n", "dE", "RMS_g", "MAX_g", "RMS_s", "MAX_s", "im_mode", "im_freq", "EigenX100", "MonConv"]]
    
    # delete the columns in del_col
    format_out = [format_out[i] for i in range(len(format_out)) if i not in del_col]
    result[0] = [result[0][i] for i in range(len(result[0])) if i not in del_col]
    
    res_cache = ["", "", ""]
    
    converge_judge = [False] * len(opt_cycle_begin)
    
    for count_cycle, idx in enumerate(range(len(opt_cycle_begin)), start=1):
        thresh = [1, 1, 1, 1, 1] 
        if idx >= len(opt_conver_rat['E_change']):
            converge_judge[idx] = False
        elif (int(abs(opt_conver_rat['E_change'][idx]) < thresh[0]) + int(opt_conver_rat['RMS_grad'][idx] < thresh[1]) + \
                int(opt_conver_rat['MAX_grad'][idx] < thresh[2]) + int(opt_conver_rat['RMS_step'][idx] < thresh[3]) + \
                int(opt_conver_rat['MAX_step'][idx] < thresh[4])) >= 4:
            converge_judge[idx] = True
        elif (int(abs(opt_conver_rat['E_change'][idx]) < thresh[0]) + int(opt_conver_rat['RMS_grad'][idx] < thresh[1]/2) + \
                int(opt_conver_rat['MAX_grad'][idx] < thresh[2]/2) ) == 3:
            converge_judge[idx] = True

    
    if not output_ratio:
        
        for key in opt_conver:
            opt_conver[key] = list(map(lambda a: a* (10**4), opt_conver[key])) 
    
    
    while len(temp_list) > 0:
        temp = temp_list.pop()
        if temp[2] == "opt_cycle_begin":
            if res_cache[0] != "":
                auxi_idx = int(res_cache[0]) - 1
                
                auxil_res = [f"{opt_conver['E_change'][auxi_idx]:.1f}", \
                    f"{opt_conver['RMS_grad'][auxi_idx]:.1f}", \
                    f"{opt_conver['MAX_grad'][auxi_idx]:.1f}", \
                    f"{opt_conver['RMS_step'][auxi_idx]:.1f}", \
                    f"{opt_conver['MAX_step'][auxi_idx]:.1f}"]

                if res_cache[2] == "":
                    converge_judge[auxi_idx] = False
                                   
                res_cache = [res_cache[0]] + auxil_res + get_res_cache(res_cache)+ \
                    [min_eigen[auxi_idx]] +["YES" if converge_judge[auxi_idx] else "NO"]

                result.append(res_cache)

                res_cache = ["", "", ""]
            else:
                pass
            res_cache[0] = str(temp[1])
        elif temp[2] == "redundant_internal":
            res_cache[1] = temp[1] if res_cache[1] == "" else res_cache[1] + " | " + temp[1]
        elif temp[2] == "imaginary_mode":
            res_cache[2] = temp[1] if res_cache[2] == "" else res_cache[2] + " | " + temp[1]
            auxi_idx = int(res_cache[0]) - 1
            
            task_type = get_task_type(lines)
            judgement_ = 1 if task_type == "TS" else 0
            if temp[1].count("-") != judgement_: # more than 1 imaginary mode
                converge_judge[auxi_idx] = False
                

    
    auxi_idx = int(res_cache[0]) - 1
    if res_cache[2] == "":
        converge_judge[auxi_idx] = False
    if auxi_idx >= len(opt_conver['E_change']):
        auxil_res = ["None"] * 5
        res_cache = [res_cache[0]] + auxil_res + get_res_cache(res_cache) + \
                    ["None"] + ["NO"]
    else:
        auxil_res = [f"{opt_conver['E_change'][auxi_idx]:.1f}", \
            f"{opt_conver['RMS_grad'][auxi_idx]:.1f}", \
            f"{opt_conver['MAX_grad'][auxi_idx]:.1f}", \
            f"{opt_conver['RMS_step'][auxi_idx]:.1f}", \
            f"{opt_conver['MAX_step'][auxi_idx]:.1f}"]
    
        res_cache = [res_cache[0]] + auxil_res + get_res_cache(res_cache) + \
            [min_eigen[auxi_idx]] + ["YES" if converge_judge[auxi_idx] else "NO"]

    result.append(res_cache)
    
    output_text = ""
    # use the number in format_out to format the output
    for item in result:
        output_text += '\t'.join([f"{item[i]:<{format_out[i]}}" for i in range(len(item))]) + '\n'
        
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(output_text)
    else:
        print(output_text)
    return output_text
    
def get_charge_mul(lines):
    text_all = "".join(lines)
    match = re.search(pattern5, text_all, re.MULTILINE | re.IGNORECASE)
    if match:
        charge, mul = match.groups()
        return int(charge), int(mul)
    else:
        raise ValueError("Charge and Multiplicity not found")
   
def get_num_of_atoms(lines):
    text_all = "".join(lines)
    match = re.search(pattern7, text_all, re.MULTILINE)
    if match:
        return int(match.groups()[0])
    else:
        raise ValueError("Number of Atoms not found")
   
    
def get_task_type(lines):
    full_text = "".join(lines)
    if opt_flag_txt in full_text:
        if ts_flag_txt in full_text:
            return "TS"
        else:
            return "Opt"
    elif freq_flag_txt in full_text:
        return "Freq"
    else:
        return "Unknown"

def get_opt_xyz_frame(lines, n_frame=-1, output_filename=''):
    # opt_cycle_begin = get_opt_cycle_begin(lines)
    opt_cycle_begin = get_cartesian_coords_begin(lines)
    charge, mul = get_charge_mul(lines)
    num_of_atoms = get_num_of_atoms(lines)
    # xyz_begin_idx = [int(item[0])-1+5 for item in opt_cycle_begin]
    xyz_begin_idx = [int(item) for item in opt_cycle_begin]
    xyz_eng_idx = [item + num_of_atoms for item in xyz_begin_idx]
    
    begin_idx_ = xyz_begin_idx[n_frame]
    end_idx_ = xyz_eng_idx[n_frame]
    
    
    
    xyz_string = f"{num_of_atoms}\nCharge: {charge} ,Mul: {mul} ,frame: {n_frame} ,type: {get_task_type(lines).lower()}\n"
    xyz_string += '\n'.join(list(map(lambda a:a.strip(), lines[begin_idx_:end_idx_])))
    
    if output_filename:
        with open(output_filename, 'w') as f:
            f.write(xyz_string)
    else:
        return xyz_string


def get_hessian_min_eigen(lines):
    text_all = "".join(lines)
    matches = re.findall(pattern8, text_all, re.MULTILINE)
    if matches:
        matches = list(map(lambda a: list(map(lambda b: f"{float(b)*100:.2f}", a)), matches))
        matches_str = list(map(lambda a: " ".join(a), matches))
        return matches_str
    else:
        raise ValueError("Lowest eigenvalues not found")

def output_freq(lines, freq_type='opt'):
    imaginary_mode = get_imaginary_mode(lines)[-1][1]
    
    num_if = imaginary_mode.count("-")
    valid_flag = None
    
    if freq_type == 'opt':
        valid_flag = True if num_if==0 else False
    elif freq_type == 'ts':
        valid_flag = True if num_if==1 else False
    else:
        raise ValueError(f"Unknown freq type: {freq_type}")
    
    print(f"frquency type:{freq_type}, imaginary_freq:  {imaginary_mode}")
    print(f"Validity:{'YES' if valid_flag else 'NO'}")
    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orcamonitor.py <filename> [-o] [-i] [-x xyz_frame] [-f freq_type]")
        sys.exit(1)

    parser = argparse.ArgumentParser(description='Process ORCA output files.')
    parser.add_argument('filename', type=str, help='The ORCA output file to process')
    parser.add_argument('-i', '--interactive', action='store_true', help='Enable interactive mode')
    parser.add_argument('-x', '--xyz', type=int, help='Extract the xyz frame, int.')
    parser.add_argument('-o', '--ongoing', action='store_true', help='never raise error if encoutered')
    parser.add_argument('-f', '--freq_type', type=str, default='opt', 
                        help='Works for Freq-Only task, to judge if if qualifies. opt or ts (default: opt)')

    args = parser.parse_args()
    filename = args.filename
    

    if os.path.exists(filename):
        pass
    else:
        print(f"{filename} not found")
        sys.exit(1)
    
    
    freq_type = (args.freq_type).lower()  
    assert freq_type in ['opt', 'ts'], f"freq_type should be 'opt' or 'ts'"
    interactive = args.interactive
    
    # # For Debug
    # filename = "PreTS_VWCVATAYLCUMCJ-UHFFFAOYSA-N.out"
    # freq_type = 'ts'
    # interactive = True
        
    with open(filename, 'r') as f:
        lines = f.readlines()
        text_all = ''.join(lines)
    

    
    def work():
        if args.xyz:
            try:
                print(get_opt_xyz_frame(lines, n_frame=args.xyz - 1))
            except:
                print(get_opt_xyz_frame(lines, n_frame=-1))
            sys.exit(0)
        
        print(f"Doing orca monitoring for {filename}")
        num_of_atoms = get_num_of_atoms(lines)
        charge, mul = get_charge_mul(lines)
        task_type = get_task_type(lines)
        print(f"Task type: {task_type.lower()}")
        print(f"Number of atoms: {num_of_atoms}  Charge: {charge}  Multiplicity: {mul}")
        
        if task_type == "Freq":
            output_freq(lines, freq_type)
            sys.exit(0)
        elif task_type == "Unknown":
            print("Unknown task type")
            sys.exit(1)
        elif task_type in ["Opt", "TS"]:
            pass
        
        opt_cycle_begin = get_opt_cycle_begin(lines)
        print(f"Optimization cycles: {len(opt_cycle_begin)}")
        
        output_ratio = True
        input_frame = None
        output_pop(lines, output_ratio=output_ratio, format_out = [2, 6, 6, 6, 6, 6, 24, 24, 18, 5])
        
        while True and interactive:
            if output_ratio:
                print("Optimization convergence printed as ratio (value/converge threshold: Normal)")
            else:
                print("Optimization convergence printed as value, X 10^4")
                
            input_frame = input("Input the frame number to extract the xyz file, input 't' to change convergence output or input 'q' to quit: ")
            if input_frame == 'q':
                break
            elif input_frame == 't':
                output_ratio = not output_ratio
                output_pop(lines, output_ratio=output_ratio, format_out = [2, 6, 6, 6, 6, 6, 24, 24, 18, 5])
                continue
            else:
                try:
                    input_frame = int(input_frame)
                except ValueError:
                    print("Invalid input")
                    continue
                res = get_opt_xyz_frame(lines, n_frame=input_frame)
                print(f"Frame {input_frame} extracted")
                print(res)
        
        sys.exit(0)
        
    if args.ongoing:
        try:
            work()
        except Exception as e:
            print(0)
            sys.exit(1)
    else:
        work()
    
    
    
