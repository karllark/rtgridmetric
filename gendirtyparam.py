
if __name__ == "__main__":
    # generate 100 input files with different random number seeds
    basefile = "input_unctest/dirty_gridmetric_slab_tau1_10_10_10.drt"

    with open(basefile) as file:
        lines = [line.rstrip() for line in file]

    for i in range(100):
        ofile = basefile.replace(".drt", f"testunc{i+1}.drt")
        print(ofile)
        with open(ofile, 'w') as f:
            for line in lines:
                if "random_num_seed" in line:
                    nline = line.replace("123456789", f"1234567{i}")
                    f.write(f"{nline}\n")
                elif "output_filebase" in line:
                    f.write(f"{line}_n{i+1}\n")
                else:
                    f.write(f"{line}\n")