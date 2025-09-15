class Stats:
    """
    Stats of a single reconciliation.
    """

    def __init__(self):
        """
        Create a new stats block with all counters initialized to zero.
        """
        self.elapsed_process_time = None
        self.elapsed_real_time = None
        self.normal_iterations = 0
        self.biconf_iterations = 0
        self.ask_parity_messages = 0
        self.ask_parity_blocks = 0
        self.ask_parity_bits = 0
        self.reply_parity_bits = 0
        self.unrealistic_efficiency = None
        self.realistic_efficiency = None
        self.infer_parity_blocks = 0
        self.estimated_corrected_bits = 0
        self.corrected_bits_error_iteration = []

        #Used more due to the Channel
        self.ask_parity_bytes = 0
        self.avg_bytes_per_block = 0
        self.avg_bytes_per_message = 0
        self.reply_parity_bytes = 0
        self.reply_parity_bytes_per_message = 0


        #Considering Bandwidth Limit
        self.total_n_chunks = 0 #Check
        self.n_chunks_per_message = 0 #Check
        self.n_chunks_per_block = 0 #Check
        self.avg_time_to_send_message = 0 #Check
        self.avg_n_blocks_per_message = 0 #Check
        self.total_time_to_send_message = 0
        self.total_time_to_send_message_math = 0

        self.total_n_chunks_received = 0
        self.n_chunks_per_message_received = 0
        self.n_chunks_per_block_received = 0
        self.avg_time_to_receive_message = 0
        self.total_time_to_receive_message = 0

        self.total_time_wait_for_receiving_messages = 0

        #Similar to Cascade CPP
        self.start_iterations_bits = 0

        self.largest_message_sent = (0,0)
        self.largest_message_received = (0,0)

