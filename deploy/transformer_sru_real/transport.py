"""Unitree HG LowCmd packing; matches SDK2 Python's 1004-byte layout."""
import struct

POLY=0x04c11db7
def _table_entry(i):
    value=i<<24
    for _ in range(8):value=((value<<1) ^ (POLY if value&0x80000000 else 0)) & 0xffffffff
    return value
TABLE=tuple(_table_entry(i) for i in range(256))


def lowcmd_crc(message):
    values=[message.mode_pr,message.mode_machine]
    for m in message.motor_cmd:
        values.extend([m.mode,m.q,m.dq,m.tau,m.kp,m.kd,m.reserve])
    values.extend(message.reserve);values.append(0)
    raw=struct.pack('<2B2x'+'B3x5fI'*35+'5I',*values)
    crc=0xffffffff
    for word, in struct.iter_unpack('<I',raw[:-4]):
        for shift in (24,16,8,0):crc=((crc<<8)&0xffffffff)^TABLE[((crc>>24)^(word>>shift))&255]
    return crc


def motor_message(mode_machine,target,kp,kd):
    from unitree_hg.msg import LowCmd
    msg=LowCmd();msg.mode_pr=0;msg.mode_machine=int(mode_machine)
    for i in range(29):
        m=msg.motor_cmd[i];m.mode=1;m.q=float(target[i]);m.dq=0.;m.tau=0.
        m.kp=float(kp[i]);m.kd=float(kd[i])
    # Unused slots 29..34 stay disabled and zeroed.
    msg.crc=lowcmd_crc(msg)
    return msg
