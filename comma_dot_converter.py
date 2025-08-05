sample_names = ['NOJP2', 'NOJP7a', 'NOJP9', 'NOJP12a', 'NOJP13', 'NOJP14']
area_numbers = [1,2,3,4,5]
laser_wavelengths = [532,785]


for sample_name in sample_names:
    for area_number in area_numbers:
        for laser_wavelength in laser_wavelengths:
            datafile = f'/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/{sample_name} Area {area_number} {laser_wavelength}_01.txt'
            new_datafile = f'/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/Fixed/{sample_name} Area {area_number} {laser_wavelength}_fixed.txt'
            with open(datafile, 'r') as inputfile:
                content = inputfile.read()

            modified_content = content.replace(',', '.')

            with open(new_datafile, 'w') as outfile:
                outfile.write(modified_content)
