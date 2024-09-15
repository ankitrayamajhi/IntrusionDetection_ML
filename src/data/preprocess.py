import pandas as pd
from sklearn import preprocessing

def preprocess_data(df):
    """Preprocess the data: adjust columns and encode categorical features."""
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'])
    df.columns = columns
    
    # Convert attack column
    df['attack'] = df['attack'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

    # Encode categorical features
    protocol_type_le = preprocessing.LabelEncoder()
    service_le = preprocessing.LabelEncoder()
    flag_le = preprocessing.LabelEncoder()

    df['protocol_type'] = protocol_type_le.fit_transform(df['protocol_type'])
    df['service'] = service_le.fit_transform(df['service'])
    df['flag'] = flag_le.fit_transform(df['flag'])

    return df
