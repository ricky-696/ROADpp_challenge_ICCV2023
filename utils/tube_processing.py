


def combine_label(agent_id, action_id, loc_id):
    agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
    action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
    loc_labels = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'OutgoBusLane', 'IncomLane', 'IncomCycLane', 'IncomBusLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'LftParking', 'rightParking']
    triplet_labels = ['Ped-MovAway-LftPav', 'Ped-MovAway-RhtPav', 'Ped-MovAway-Jun', 'Ped-MovTow-LftPav', 'Ped-MovTow-RhtPav', 'Ped-MovTow-Jun', 'Ped-Mov-OutgoLane', 'Ped-Mov-Pav', 'Ped-Mov-RhtPav', 'Ped-Stop-OutgoLane', 'Ped-Stop-Pav', 'Ped-Stop-LftPav', 'Ped-Stop-RhtPav', 'Ped-Stop-BusStop', 'Ped-Wait2X-RhtPav', 'Ped-Wait2X-Jun', 'Ped-XingFmLft-Jun', 'Ped-XingFmRht-Jun', 'Ped-XingFmRht-xing', 'Ped-Xing-Jun', 'Ped-PushObj-LftPav', 'Ped-PushObj-RhtPav', 'Car-MovAway-VehLane', 'Car-MovAway-OutgoLane', 'Car-MovAway-Jun', 'Car-MovTow-VehLane', 'Car-MovTow-IncomLane', 'Car-MovTow-Jun', 'Car-Brake-VehLane', 'Car-Brake-OutgoLane', 'Car-Brake-Jun', 'Car-Stop-VehLane', 'Car-Stop-OutgoLane', 'Car-Stop-IncomLane', 'Car-Stop-Jun', 'Car-Stop-parking', 'Car-IncatLft-VehLane', 'Car-IncatLft-OutgoLane', 'Car-IncatLft-IncomLane', 'Car-IncatLft-Jun', 'Car-IncatRht-VehLane', 'Car-IncatRht-OutgoLane', 'Car-IncatRht-IncomLane', 'Car-IncatRht-Jun', 'Car-HazLit-IncomLane', 'Car-TurLft-VehLane', 'Car-TurLft-Jun', 'Car-TurRht-Jun', 'Car-MovRht-OutgoLane', 'Car-MovLft-VehLane', 'Car-MovLft-OutgoLane', 'Car-XingFmLft-Jun', 'Car-XingFmRht-Jun', 'Cyc-MovAway-OutgoCycLane', 'Cyc-MovAway-RhtPav', 'Cyc-MovTow-IncomLane', 'Cyc-MovTow-RhtPav', 'MedVeh-MovAway-VehLane', 'MedVeh-MovAway-OutgoLane', 'MedVeh-MovAway-Jun', 'MedVeh-MovTow-IncomLane', 'MedVeh-MovTow-Jun', 'MedVeh-Brake-VehLane', 'MedVeh-Brake-OutgoLane', 'MedVeh-Brake-Jun', 'MedVeh-Stop-VehLane', 'MedVeh-Stop-OutgoLane', 'MedVeh-Stop-IncomLane', 'MedVeh-Stop-Jun', 'MedVeh-Stop-parking', 'MedVeh-IncatLft-IncomLane', 'MedVeh-IncatRht-Jun', 'MedVeh-TurRht-Jun', 'MedVeh-XingFmLft-Jun', 'MedVeh-XingFmRht-Jun', 'LarVeh-MovAway-VehLane', 'LarVeh-MovTow-IncomLane', 'LarVeh-Stop-VehLane', 'LarVeh-Stop-Jun', 'Bus-MovAway-OutgoLane', 'Bus-MovTow-IncomLane', 'Bus-Stop-VehLane', 'Bus-Stop-OutgoLane', 'Bus-Stop-IncomLane', 'Bus-Stop-Jun', 'Bus-HazLit-OutgoLane']

    tube_tri_label = agent_labels[agent_id] + '-' + action_labels[action_id] + '-' + loc_labels[loc_id]

    if tube_tri_label in triplet_labels:
        return triplet_labels.index(tube_tri_label)
    else:
        return -1


if __name__ == '__main__':
    idx = combine_label(0, 3, 8)
    print(idx)