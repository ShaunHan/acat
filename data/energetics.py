gas_potential_energies = {'CH4': -24.039, 'OH2': -14.168, 'H2': -6.989}

adsorbate_potential_energies = {'H': gas_potential_energies['H2'] / 2, 
                                'C': gas_potential_energies['CH4'] - 2 * gas_potential_energies['H2'],
                                'O': gas_potential_energies['OH2'] - gas_potential_energies['H2'],
                                'CH': gas_potential_energies['CH4'] - 1.5 * gas_potential_energies['H2'],
                                'CO': gas_potential_energies['CH4'] + gas_potential_energies['OH2'] - 3 * gas_potential_energies['H2'],
                                'COH': gas_potential_energies['CH4'] + gas_potential_energies['OH2'] - 2.5 * gas_potential_energies['H2'],
                                'CHO': gas_potential_energies['CH4'] + gas_potential_energies['OH2'] - 2.5 * gas_potential_energies['H2'],}

label_free_energy_corrections = {'1H': 0.787, '2H': 0.787, '3H': 0.910, '4H': 0.910, '6H': 0.922, '7H': 0.901, '10H': 0.913, '11H': 0.903,
                                 '3C': -2.331, '4C': -2.331, '6C': -2.277, '7C': -2.285, '10C': -2.292, '11C': -2.280, '14C': -2.271, '21C': -2.277,
                                 '6O': -0.195, '7O': -0.206, '10O': -0.208, '11O': -0.215,
                                 '3CH': -1.293, '4CH': -1.293, '6CH': -1.292, '7CH': -1.282, '10CH': -1.301, '11CH': -1.293,
                                 '1CO': -2.828, '2CO': -2.750, '3CO': -2.697, '4CO': -2.697, '6CO': -2.624, '7CO': -2.636, '10CO': -2.679, '11CO': -2.619,
                                 '6COH': -1.711, '7COH': -1.580, '10COH': -1.624, '11COH': -1.592,
                                 '6CHO': -1.576, '7CHO': -1.566, '10CHO': -1.619, '11CHO': -1.565,}


