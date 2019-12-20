import numpy as np
import os
import time

# From phconvert I get the following functions

def load_pt3(filename, ovcfunc=None):
    """Load data from a PicoQuant .pt3 file.
    Arguments:
        filename (string): the path of the PT3 file to be loaded.
        ovcfunc (function or None): function to use for overflow/rollover
            correction of timestamps. If None, it defaults to the
            fastest available implementation for the current machine.
    Returns:
        A tuple of timestamps, detectors, nanotimes (integer arrays) and a
        dictionary with metadata containing at least the keys
        'timestamps_unit' and 'nanotimes_unit'.
    """
    assert os.path.isfile(filename), "File '%s' not found." % filename

    t3records, timestamps_unit, nanotimes_unit, meta = pt3_reader(filename)
    detectors, timestamps, nanotimes = process_t3records(
        t3records, time_bit=16, dtime_bit=12, ch_bit=4, special_bit=False,
        ovcfunc=ovcfunc)
    acquisition_duration = meta['header']['AcquisitionTime'][0] * 1e-3
    ctime_t = time.strptime(meta['header']['FileTime'][0].decode(),
                            "%d/%m/%y %H:%M:%S")
    creation_time = time.strftime("%Y-%m-%d %H:%M:%S", ctime_t)
    meta.update({'timestamps_unit': timestamps_unit,
                 'nanotimes_unit': nanotimes_unit,
                 'acquisition_duration': acquisition_duration,
                 'laser_repetition_rate': meta['ttmode']['InpRate0'],
                 'software': meta['header']['CreatorName'][0].decode(),
                 'software_version': meta['header']['CreatorVersion'][0].decode(),
                 'creation_time': creation_time,
                 'hardware_name': meta['header']['Ident'][0].decode(),
                 })
    return timestamps, detectors, nanotimes, meta

def pt3_reader(filename):
    """Load raw t3 records and metadata from a PT3 file.
    """
    with open(filename, 'rb') as f:
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Binary file header
        header_dtype = np.dtype([
            ('Ident',             'S16'   ),
            ('FormatVersion',     'S6'    ),
            ('CreatorName',       'S18'   ),
            ('CreatorVersion',    'S12'   ),
            ('FileTime',          'S18'   ),
            ('CRLF',              'S2'    ),
            ('Comment',           'S256'  ),
            ('NumberOfCurves',    'int32' ),
            ('BitsPerRecord',     'int32' ),   # bits in each T3 record
            ('RoutingChannels',   'int32' ),
            ('NumberOfBoards',    'int32' ),
            ('ActiveCurve',       'int32' ),
            ('MeasurementMode',   'int32' ),
            ('SubMode',           'int32' ),
            ('RangeNo',           'int32' ),
            ('Offset',            'int32' ),
            ('AcquisitionTime',   'int32' ),   # in ms
            ('StopAt',            'uint32'),
            ('StopOnOvfl',        'int32' ),
            ('Restart',           'int32' ),
            ('DispLinLog',        'int32' ),
            ('DispTimeAxisFrom',  'int32' ),
            ('DispTimeAxisTo',    'int32' ),
            ('DispCountAxisFrom', 'int32' ),
            ('DispCountAxisTo',   'int32' ),
        ])
        header = np.fromfile(f, dtype=header_dtype, count=1)

        if header['FormatVersion'][0] != b'2.0':
            raise IOError(("Format '%s' not supported. "
                           "Only valid format is '2.0'.") % \
                           header['FormatVersion'][0])

        dispcurve_dtype = np.dtype([
            ('DispCurveMapTo', 'int32'),
            ('DispCurveShow',  'int32')])
        dispcurve = np.fromfile(f, dispcurve_dtype, count=8)

        params_dtype = np.dtype([
            ('ParamStart', 'f4'),
            ('ParamStep',  'f4'),
            ('ParamEnd',   'f4')])
        params = np.fromfile(f, params_dtype, count=3)

        repeat_dtype = np.dtype([
            ('RepeatMode',      'int32'),
            ('RepeatsPerCurve', 'int32'),
            ('RepeatTime',      'int32'),
            ('RepeatWaitTime',  'int32'),
            ('ScriptName',      'S20'  )])
        repeatgroup = np.fromfile(f, repeat_dtype, count=1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Hardware information header
        hw_dtype = np.dtype([
            ('HardwareIdent',   'S16'  ),
            ('HardwarePartNo',  'S8'   ),
            ('HardwareSerial',  'int32'),
            ('SyncDivider',     'int32'),
            ('CFDZeroCross0',   'int32'),
            ('CFDLevel0',       'int32'),
            ('CFDZeroCross1',   'int32'),
            ('CFDLevel1',       'int32'),
            ('Resolution',      'f4'),
            ('RouterModelCode', 'int32'),
            ('RouterEnabled',   'int32')])
        hardware = np.fromfile(f, hw_dtype, count=1)

        rtr_dtype = np.dtype([
            ('InputType',       'int32'),
            ('InputLevel',      'int32'),
            ('InputEdge',       'int32'),
            ('CFDPresent',      'int32'),
            ('CFDLevel',        'int32'),
            ('CFDZCross',       'int32')])
        router = np.fromfile(f, rtr_dtype, count=4)

        # Time tagging mode specific header
        ttmode_dtype = np.dtype([
            ('ExtDevices',      'int32' ),
            ('Reserved1',       'int32' ),
            ('Reserved2',       'int32' ),
            ('InpRate0',        'int32' ),
            ('InpRate1',        'int32' ),
            ('StopAfter',       'int32' ),
            ('StopReason',      'int32' ),
            ('nRecords',        'int32' ),
            ('ImgHdrSize',      'int32')])
        ttmode = np.fromfile(f, ttmode_dtype, count=1)

        # Special header for imaging. How many of the following ImgHdr
        # array elements are actually present in the file is indicated by
        # ImgHdrSize above.
        ImgHdr = np.fromfile(f, dtype='int32', count=ttmode['ImgHdrSize'][0])

        # The remainings are all T3 records
        t3records = np.fromfile(f, dtype='uint32', count=ttmode['nRecords'][0])

        timestamps_unit = 1./ttmode['InpRate0']
        nanotimes_unit = 1e-9*hardware['Resolution']

        metadata = dict(header=header, dispcurve=dispcurve, params=params,
                        repeatgroup=repeatgroup, hardware=hardware,
                        router=router, ttmode=ttmode, imghdr=ImgHdr)
        return t3records, timestamps_unit, nanotimes_unit, metadata

def process_t3records(t3records, time_bit=10, dtime_bit=15,
                      ch_bit=6, special_bit=True, ovcfunc=None):
    """Extract the different fields from the raw t3records array.
    The input array of t3records is an array of "records" (a C struct).
    It packs all the information of each detected photons. This function
    decodes the different fields and returns 3 arrays
    containing the timestamps (i.e. macro-time or number of sync,
    few-ns resolution), the nanotimes (i.e. the micro-time or TCSPC time,
    ps resolution) and the detectors.
    t3records have these fields (in little-endian order)::
        | Optional special bit | detectors | nanotimes | timestamps |
          MSB                                                   LSB
    Bit allocation of these fields, starting from the MSB:
    - **special bit**: 1 bit if `special_bit = True` (default), else no special bit.
    - **channel**: default 6 bit, (argument `ch_bit`), detector or special marker
    - **nanotimes**: default 15 bit (argument `dtime_bit`), nanotimes (TCSPC time)
    - **timestamps**: default 10 bit, (argument `time_bit`), the timestamps (macro-time)

    **Timestamps**: The returned timestamps are overflow-corrected, and therefore
    should be monotonically increasing. Each overflow event is marked by
    a special detector (or a special bit) and this information is used for
    the correction. These overflow "events" **are not removed** in the returned
    arrays resulting in spurious detectors. This choice has been made for
    safety (you can always go and check where there was an overflow) and for
    efficiency (removing a few elements requires allocating a new array that
    is potentially expensive for big data files). Under normal usage the
    additional detectors take negligible space and can be safely ignored.
    Arguments:
        t3records (array): raw array of t3records as saved in the
            PicoQuant file.
        time_bit (int): number of bits in the t3record used for timestamps
            (or macro-time).
        dtime_bit (int): number of bits in the t3record used for the nanotime
            (TCSPC time or micro-time)
        ch_bit (int): number of bits in the t3record used for the detector
            number.
        special_bit (bool): if True the t3record contains a special bit
            for overflow correction.
            This special bit will become the MSB in the returned detectors
            array. If False, it assumes no special bit in the t3record.
        ovcfunc (function or None): function to perform overflow correction
            of timestamps. If None use the default function. The default
            function is the numba-accelerated version is numba is installed
            otherwise it is function using plain numpy.
    Returns:
        A 3-element tuple containing the following 1D arrays (all of the same
        length):
        - **timestamps** (*array of int64*): the macro-time (or number of sync)
          of each photons after overflow correction. Units are specified in
          the file header.
        - **nanotimes** (*array of uint16*): the micro-time (TCSPC time), i.e.
          the time lag between the photon detection and the previous laser
          sync. Units (i.e. the bin width) are specified in the file header.
        - **detectors** (*arrays of uint8*): detector number. When
          `special_bit = True` the highest bit in `detectors` will be
          the special bit.
    """

    """
    Notes on detectors:
        The bit allocation in the record is, starting from the MSB::
            special: 1
            channel: 6
            dtime: 15
            nsync: 10

        If the special bit is clear, it's a regular event record.
        If the special bit is set, the following interpretation of
        the channel code is given:
        - code 63 (all bits ones) identifies a sync count overflow,
          increment the sync count overflow accumulator. For
          HydraHarp V1 ($00010304) it means always one overflow.
          For all other types the number of overflows can be read from nsync value.
        - codes from 1 to 15 identify markers, the individual bits are external markers.
        If detectors is above 64 then it is a special record.

            detectors==127 =>overflow
            detectors==65 => Marker 1 event
            detectors==66 => Marker 2 event
            ...
            detectors==79 => Marker 15 event
        else if
            detectors==0 => regular event regular detector 0
            detectors==1 => regular event regular detector 1
            detectors==2 => regular event regular detector 2
            ...
    """

    if special_bit:
        ch_bit += 1
    assert ch_bit <= 8
    assert time_bit <= 16
    assert time_bit + dtime_bit + ch_bit == 32

    detectors = np.bitwise_and(
        np.right_shift(t3records, time_bit + dtime_bit),
        2**ch_bit - 1).astype('uint8')
    nanotimes = np.bitwise_and(
        np.right_shift(t3records, time_bit),
        2**dtime_bit - 1).astype('uint16')

    dt = np.dtype([('low16', 'uint16'), ('high16', 'uint16')])
    t3records_low16 = np.frombuffer(t3records, dt)['low16']     # View
    timestamps = t3records_low16.astype(np.int64)               # Copy
    np.bitwise_and(timestamps, 2**time_bit - 1, out=timestamps)

    overflow_ch = 2**ch_bit - 1
    overflow = 2**time_bit
    if ovcfunc is None:
        ovcfunc = _correct_overflow
    ovcfunc(timestamps, detectors, overflow_ch, overflow)
    return detectors, timestamps, nanotimes

def _correct_overflow(timestamps, detectors, overflow_ch, overflow):
    """Apply overflow correction when each overflow has a special timestamp.
    """
    index_overflows = np.where((detectors == overflow_ch))[0]
    for n, (idx1, idx2) in enumerate(zip(index_overflows[:-1],
                                         index_overflows[1:])):
        timestamps[idx1:idx2] += (n + 1)*overflow
    timestamps[idx2:] += (n + 2)*overflow


# Define my functions
def extract(path):

    data = load_pt3(path)
    new_data = list(zip(data[0], data[1], data[2]))
    timestamps_unit = data[-1]['timestamps_unit']
    nanotimes_unit = data[-1]['nanotimes_unit']

    return new_data, timestamps_unit, nanotimes_unit


def folder_walker(path):

    for i in os.listdir(path):
        if i.endswith('.pt3'):
            photons, microtime_unit, nanotime_unit = extract(os.path.join(path,i))
            np.savetxt(path + '/' + i[:-4] + '.csv', photons, delimiter = '\t', header = f'{microtime_unit},     ,      {nanotime_unit}\nMicrotime,    Channel,    Nanotime')


if __name__ == '__main__':
    path = input('Please enter the path to the folder where the pt3 files are located: ')
    folder_walker(path)
