#!/usr/bin/env python3
"""
Network Topology Discovery via SNMP
Descubre topología usando SNMP exclusivamente (sin scapy)
"""

import sys
import socket
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from collections import defaultdict
import ipaddress

try:
    from pysnmp.hlapi import *

    SNMP_AVAILABLE = True
except ImportError:
    SNMP_AVAILABLE = False
    print("❌ pysnmp no instalado. Instalar con: pip install pysnmp")
    sys.exit(1)


@dataclass
class SNMPDevice:
    """Dispositivo descubierto via SNMP"""
    ip: str
    hostname: str = ""
    sysDescr: str = ""
    sysObjectID: str = ""
    sysUpTime: str = ""
    manufacturer: str = ""
    model: str = ""
    device_type: str = "unknown"
    interfaces: Dict[int, 'Interface'] = field(default_factory=dict)
    neighbors: Dict[int, List['Neighbor']] = field(default_factory=dict)
    arp_table: List['ArpEntry'] = field(default_factory=list)
    routing_table: List['Route'] = field(default_factory=list)


@dataclass
class Interface:
    """Interfaz de red"""
    ifIndex: int
    ifDescr: str = ""
    ifType: str = ""
    ifMtu: int = 0
    ifSpeed: int = 0
    ifPhysAddress: str = ""
    ifAdminStatus: int = 0  # 1=up, 2=down, 3=testing
    ifOperStatus: int = 0  # 1=up, 2=down, 3=testing, 4=unknown
    ifInOctets: int = 0
    ifOutOctets: int = 0
    ipAddress: str = ""


@dataclass
class Neighbor:
    """Vecino descubierto via LLDP/CDP"""
    remote_chassis_id: str = ""
    remote_port_id: str = ""
    remote_system_name: str = ""
    remote_system_descr: str = ""


@dataclass
class ArpEntry:
    """Entrada de tabla ARP"""
    ip_address: str
    mac_address: str
    interface_index: int


@dataclass
class Route:
    """Entrada de tabla de rutas"""
    destination: str
    mask: str
    next_hop: str
    interface_index: int
    metric: int


class SNMPDiscovery:
    """Descubrimiento de topología usando SNMP"""

    # OIDs estándar MIB-II
    OID_SYSTEM = {
        'sysDescr': '1.3.6.1.2.1.1.1.0',
        'sysObjectID': '1.3.6.1.2.1.1.2.0',
        'sysUpTime': '1.3.6.1.2.1.1.3.0',
        'sysContact': '1.3.6.1.2.1.1.4.0',
        'sysName': '1.3.6.1.2.1.1.5.0',
        'sysLocation': '1.3.6.1.2.1.1.6.0',
    }

    # OIDs de interfaces
    OID_INTERFACES = {
        'ifIndex': '1.3.6.1.2.1.2.2.1.1',
        'ifDescr': '1.3.6.1.2.1.2.2.1.2',
        'ifType': '1.3.6.1.2.1.2.2.1.3',
        'ifMtu': '1.3.6.1.2.1.2.2.1.4',
        'ifSpeed': '1.3.6.1.2.1.2.2.1.5',
        'ifPhysAddress': '1.3.6.1.2.1.2.2.1.6',
        'ifAdminStatus': '1.3.6.1.2.1.2.2.1.7',
        'ifOperStatus': '1.3.6.1.2.1.2.2.1.8',
    }

    # OIDs de tabla ARP (ipNetToMediaTable)
    OID_ARP = {
        'ipNetToMediaPhysAddress': '1.3.6.1.2.1.4.22.1.2',
        'ipNetToMediaNetAddress': '1.3.6.1.2.1.4.22.1.3',
    }

    # OIDs de tabla de rutas
    OID_ROUTES = {
        'ipRouteDest': '1.3.6.1.2.1.4.21.1.1',
        'ipRouteNextHop': '1.3.6.1.2.1.4.21.1.7',
        'ipRouteMetric1': '1.3.6.1.2.1.4.21.1.3',
    }

    # OIDs de LLDP (si está disponible)
    OID_LLDP = {
        'lldpRemChassisId': '1.0.8802.1.1.2.1.4.1.1.5',
        'lldpRemPortId': '1.0.8802.1.1.2.1.4.1.1.7',
        'lldpRemSysName': '1.0.8802.1.1.2.1.4.1.1.9',
    }

    def __init__(self, seed_ips: List[str], community: str = "public", timeout: int = 2):
        """
        Args:
            seed_ips: Lista de IPs iniciales para comenzar el descubrimiento
            community: Comunidad SNMP (default: public)
            timeout: Timeout en segundos para consultas SNMP
        """
        self.seed_ips = seed_ips
        self.community = community
        self.timeout = timeout
        self.devices: Dict[str, SNMPDevice] = {}
        self.discovered_ips: Set[str] = set()

    def discover_recursive(self, max_depth: int = 3):
        """
        Descubre topología recursivamente usando tablas ARP y rutas

        Args:
            max_depth: Profundidad máxima de recursión
        """
        print("=" * 80)
        print("DESCUBRIMIENTO DE TOPOLOGÍA VIA SNMP")
        print("=" * 80)
        print(f"IPs iniciales: {', '.join(self.seed_ips)}")
        print(f"Comunidad SNMP: {self.community}")
        print(f"Profundidad máxima: {max_depth}\n")

        # Nivel 0: Seeds
        to_discover = set(self.seed_ips)

        for depth in range(max_depth):
            print(f"\n{'=' * 80}")
            print(f"NIVEL {depth + 1} - Dispositivos a explorar: {len(to_discover)}")
            print(f"{'=' * 80}")

            if not to_discover:
                print("  No hay más dispositivos por explorar")
                break

            # Descubrir dispositivos en este nivel
            newly_discovered = set()

            for ip in list(to_discover):
                if ip in self.discovered_ips:
                    continue

                print(f"\n[{ip}] Explorando...")

                device = self.query_device(ip)

                if device:
                    self.devices[ip] = device
                    self.discovered_ips.add(ip)

                    print(f"  ✓ {device.hostname or ip} ({device.device_type})")
                    print(f"    Interfaces: {len(device.interfaces)}")
                    print(f"    Entradas ARP: {len(device.arp_table)}")
                    print(f"    Vecinos LLDP: {sum(len(v) for v in device.neighbors.values())}")

                    # Extraer nuevas IPs de tablas ARP
                    for arp_entry in device.arp_table:
                        new_ip = arp_entry.ip_address
                        if new_ip not in self.discovered_ips and self._is_valid_ip(new_ip):
                            newly_discovered.add(new_ip)

                    # Extraer nuevas IPs de tabla de rutas
                    for route in device.routing_table:
                        next_hop = route.next_hop
                        if next_hop not in self.discovered_ips and self._is_valid_ip(next_hop):
                            if next_hop not in ['0.0.0.0', '127.0.0.1']:
                                newly_discovered.add(next_hop)
                else:
                    print(f"  ✗ No responde a SNMP o timeout")

            # Preparar siguiente nivel
            to_discover = newly_discovered

            if newly_discovered:
                print(f"\n  → Descubiertos {len(newly_discovered)} dispositivos nuevos para el siguiente nivel")

        print(f"\n{'=' * 80}")
        print(f"DESCUBRIMIENTO COMPLETADO")
        print(f"{'=' * 80}")
        print(f"Total dispositivos: {len(self.devices)}")

        return self.devices

    def query_device(self, ip: str) -> Optional[SNMPDevice]:
        """Consulta información completa de un dispositivo via SNMP"""
        device = SNMPDevice(ip=ip)

        try:
            # 1. Información del sistema
            if not self._query_system_info(ip, device):
                return None  # Si no responde a consultas básicas, no es un dispositivo SNMP

            # 2. Identificar tipo de dispositivo
            device.device_type = self._classify_device(device)

            # 3. Interfaces
            self._query_interfaces(ip, device)

            # 4. Tabla ARP
            self._query_arp_table(ip, device)

            # 5. Tabla de rutas
            self._query_routing_table(ip, device)

            # 6. Vecinos LLDP (si está disponible)
            self._query_lldp_neighbors(ip, device)

            return device

        except Exception as e:
            print(f"  ⚠️  Error consultando {ip}: {e}")
            return None

    def _query_system_info(self, ip: str, device: SNMPDevice) -> bool:
        """Consulta información básica del sistema"""
        try:
            for name, oid in self.OID_SYSTEM.items():
                iterator = getCmd(
                    SnmpEngine(),
                    CommunityData(self.community),
                    UdpTransportTarget((ip, 161), timeout=self.timeout, retries=1),
                    ContextData(),
                    ObjectType(ObjectIdentity(oid))
                )

                errorIndication, errorStatus, errorIndex, varBinds = next(iterator)

                if errorIndication or errorStatus:
                    if name == 'sysName':  # Si falla sysName, el dispositivo no responde
                        return False
                    continue

                for varBind in varBinds:
                    value = varBind[1].prettyPrint()

                    if name == 'sysName':
                        device.hostname = value
                    elif name == 'sysDescr':
                        device.sysDescr = value
                    elif name == 'sysObjectID':
                        device.sysObjectID = value
                    elif name == 'sysUpTime':
                        device.sysUpTime = value

            return True

        except Exception as e:
            return False

    def _classify_device(self, device: SNMPDevice) -> str:
        """Clasifica el tipo de dispositivo según sysDescr y sysObjectID"""
        descr = device.sysDescr.lower()
        oid = device.sysObjectID.lower()

        # Routers
        if any(x in descr for x in ['router', 'ios', 'junos', 'mikrotik']):
            return 'router'

        # Switches
        if any(x in descr for x in ['switch', 'catalyst', 'procurve']):
            return 'switch'

        # Firewalls
        if any(x in descr for x in ['firewall', 'fortigate', 'palo alto', 'checkpoint']):
            return 'firewall'

        # Servidores
        if any(x in descr for x in ['linux', 'windows', 'ubuntu', 'centos', 'debian']):
            return 'server'

        # Por OID
        if '1.3.6.1.4.1.9' in oid:  # Cisco
            if 'switch' in descr:
                return 'switch'
            else:
                return 'router'

        return 'network_device'

    def _query_interfaces(self, ip: str, device: SNMPDevice):
        """Consulta tabla de interfaces"""
        try:
            # Walk de ifTable
            for (errorIndication, errorStatus, errorIndex, varBinds) in nextCmd(
                    SnmpEngine(),
                    CommunityData(self.community),
                    UdpTransportTarget((ip, 161), timeout=self.timeout, retries=1),
                    ContextData(),
                    ObjectType(ObjectIdentity('1.3.6.1.2.1.2.2.1')),  # ifTable
                    lexicographicMode=False
            ):
                if errorIndication or errorStatus:
                    break

                for varBind in varBinds:
                    oid_str = str(varBind[0])
                    value = varBind[1]

                    # Extraer ifIndex del OID
                    parts = oid_str.split('.')
                    if len(parts) < 11:
                        continue

                    column = int(parts[10])  # Columna de ifTable
                    ifIndex = int(parts[11])  # ifIndex

                    if ifIndex not in device.interfaces:
                        device.interfaces[ifIndex] = Interface(ifIndex=ifIndex)

                    iface = device.interfaces[ifIndex]

                    # Mapear columnas
                    if column == 2:  # ifDescr
                        iface.ifDescr = str(value)
                    elif column == 3:  # ifType
                        iface.ifType = str(value)
                    elif column == 5:  # ifSpeed
                        iface.ifSpeed = int(value)
                    elif column == 6:  # ifPhysAddress
                        iface.ifPhysAddress = value.prettyPrint()
                    elif column == 7:  # ifAdminStatus
                        iface.ifAdminStatus = int(value)
                    elif column == 8:  # ifOperStatus
                        iface.ifOperStatus = int(value)

        except Exception as e:
            pass

    def _query_arp_table(self, ip: str, device: SNMPDevice):
        """Consulta tabla ARP del dispositivo"""
        try:
            arp_entries = {}

            # Walk de ipNetToMediaTable
            for (errorIndication, errorStatus, errorIndex, varBinds) in nextCmd(
                    SnmpEngine(),
                    CommunityData(self.community),
                    UdpTransportTarget((ip, 161), timeout=self.timeout, retries=1),
                    ContextData(),
                    ObjectType(ObjectIdentity('1.3.6.1.2.1.4.22.1')),  # ipNetToMediaTable
                    lexicographicMode=False
            ):
                if errorIndication or errorStatus:
                    break

                for varBind in varBinds:
                    oid_str = str(varBind[0])
                    value = varBind[1]

                    parts = oid_str.split('.')
                    if len(parts) < 14:
                        continue

                    column = int(parts[10])
                    ifIndex = int(parts[11])
                    ip_address = '.'.join(parts[12:16])

                    key = (ifIndex, ip_address)

                    if key not in arp_entries:
                        arp_entries[key] = ArpEntry(
                            ip_address=ip_address,
                            mac_address="",
                            interface_index=ifIndex
                        )

                    if column == 2:  # PhysAddress (MAC)
                        mac = value.prettyPrint()
                        arp_entries[key].mac_address = mac

            device.arp_table = list(arp_entries.values())

        except Exception as e:
            pass

    def _query_routing_table(self, ip: str, device: SNMPDevice):
        """Consulta tabla de rutas"""
        try:
            routes = {}

            for (errorIndication, errorStatus, errorIndex, varBinds) in nextCmd(
                    SnmpEngine(),
                    CommunityData(self.community),
                    UdpTransportTarget((ip, 161), timeout=self.timeout, retries=1),
                    ContextData(),
                    ObjectType(ObjectIdentity('1.3.6.1.2.1.4.21.1')),  # ipRouteTable
                    lexicographicMode=False
            ):
                if errorIndication or errorStatus:
                    break

                for varBind in varBinds:
                    oid_str = str(varBind[0])
                    value = varBind[1]

                    parts = oid_str.split('.')
                    if len(parts) < 14:
                        continue

                    column = int(parts[10])
                    dest_ip = '.'.join(parts[11:15])

                    if dest_ip not in routes:
                        routes[dest_ip] = Route(
                            destination=dest_ip,
                            mask="",
                            next_hop="",
                            interface_index=0,
                            metric=0
                        )

                    if column == 7:  # ipRouteNextHop
                        routes[dest_ip].next_hop = str(value)
                    elif column == 3:  # ipRouteMetric1
                        routes[dest_ip].metric = int(value)

            device.routing_table = list(routes.values())

        except Exception as e:
            pass

    def _query_lldp_neighbors(self, ip: str, device: SNMPDevice):
        """Consulta vecinos LLDP"""
        try:
            neighbors = defaultdict(list)

            for (errorIndication, errorStatus, errorIndex, varBinds) in nextCmd(
                    SnmpEngine(),
                    CommunityData(self.community),
                    UdpTransportTarget((ip, 161), timeout=self.timeout, retries=1),
                    ContextData(),
                    ObjectType(ObjectIdentity('1.0.8802.1.1.2.1.4.1.1')),  # lldpRemTable
                    lexicographicMode=False
            ):
                if errorIndication or errorStatus:
                    break

                for varBind in varBinds:
                    oid_str = str(varBind[0])
                    value = varBind[1]

                    # Parsear OID para extraer índices
                    # Format: ...1.4.1.1.{column}.{timeMark}.{localPortNum}.{remIndex}
                    parts = oid_str.split('.')
                    if len(parts) < 16:
                        continue

                    column = int(parts[11])
                    localPortNum = int(parts[13])
                    remIndex = int(parts[14])

                    key = (localPortNum, remIndex)

                    if key not in neighbors:
                        neighbors[localPortNum].append(Neighbor())

                    neighbor = neighbors[localPortNum][-1]

                    if column == 5:  # lldpRemChassisId
                        neighbor.remote_chassis_id = value.prettyPrint()
                    elif column == 7:  # lldpRemPortId
                        neighbor.remote_port_id = value.prettyPrint()
                    elif column == 9:  # lldpRemSysName
                        neighbor.remote_system_name = str(value)

            device.neighbors = dict(neighbors)

        except Exception as e:
            pass

    def _is_valid_ip(self, ip: str) -> bool:
        """Valida si una IP es válida y alcanzable"""
        try:
            ipaddress.ip_address(ip)
            # Excluir IPs especiales
            if ip in ['0.0.0.0', '255.255.255.255', '127.0.0.1']:
                return False
            if ip.startswith('127.') or ip.startswith('169.254.'):
                return False
            return True
        except:
            return False

    def export_to_json(self) -> Dict:
        """Exporta topología a formato JSON"""
        devices_dict = {}
        links = []

        for ip, device in self.devices.items():
            devices_dict[ip] = {
                'ip': ip,
                'hostname': device.hostname,
                'type': device.device_type,
                'manufacturer': device.manufacturer,
                'model': device.model,
                'os': device.sysDescr,
                'uptime': device.sysUpTime,
                'interfaces': {
                    idx: {
                        'name': iface.ifDescr,
                        'mac': iface.ifPhysAddress,
                        'status': 'up' if iface.ifOperStatus == 1 else 'down',
                        'speed': iface.ifSpeed,
                    }
                    for idx, iface in device.interfaces.items()
                }
            }

            # Crear enlaces desde tabla ARP
            for arp_entry in device.arp_table:
                target_ip = arp_entry.ip_address
                if target_ip in self.devices:
                    links.append({
                        'source': ip,
                        'target': target_ip,
                        'protocol': 'arp',
                        'mac': arp_entry.mac_address,
                    })

            # Crear enlaces desde vecinos LLDP
            for port_idx, neighbors in device.neighbors.items():
                for neighbor in neighbors:
                    # Buscar el dispositivo vecino por hostname
                    for target_ip, target_dev in self.devices.items():
                        if target_dev.hostname == neighbor.remote_system_name:
                            links.append({
                                'source': ip,
                                'target': target_ip,
                                'protocol': 'lldp',
                                'local_port': device.interfaces.get(port_idx, Interface(ifIndex=port_idx)).ifDescr,
                                'remote_port': neighbor.remote_port_id,
                            })
                            break

        # Eliminar enlaces duplicados
        unique_links = []
        seen = set()
        for link in links:
            key = tuple(sorted([link['source'], link['target']]))
            if key not in seen:
                seen.add(key)
                unique_links.append(link)

        return {
            'devices': devices_dict,
            'links': unique_links
        }

    def print_summary(self):
        """Imprime resumen de topología"""
        print("\n" + "=" * 80)
        print("RESUMEN DE TOPOLOGÍA")
        print("=" * 80)

        type_count = defaultdict(int)
        for device in self.devices.values():
            type_count[device.device_type] += 1

        print(f"\nTotal dispositivos: {len(self.devices)}")
        print("\nPor tipo:")
        for dev_type, count in sorted(type_count.items()):
            print(f"  {dev_type:20s}: {count}")

        print("\n" + "=" * 80)
        print("DISPOSITIVOS DETECTADOS")
        print("=" * 80)

        for ip, device in sorted(self.devices.items()):
            print(f"\n{ip:15s} - {device.hostname or '(sin nombre)'} ({device.device_type})")
            if device.sysDescr:
                print(f"  Descripción: {device.sysDescr[:80]}")
            print(f"  Interfaces: {len(device.interfaces)}")
            print(f"  Tabla ARP: {len(device.arp_table)} entradas")
            print(f"  Vecinos LLDP: {sum(len(v) for v in device.neighbors.values())}")


def main():
    """Función principal"""
    import argparse

    parser = argparse.ArgumentParser(description='Descubrimiento de topología via SNMP')
    parser.add_argument(
        'seeds',
        nargs='+',
        help='IPs iniciales para comenzar el descubrimiento (ej: 192.168.1.1)'
    )
    parser.add_argument(
        '--community',
        default='public',
        help='Comunidad SNMP (default: public)'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=3,
        help='Profundidad máxima de descubrimiento (default: 3)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=2,
        help='Timeout en segundos para consultas SNMP (default: 2)'
    )
    parser.add_argument(
        '--output',
        default='network_topology_snmp.json',
        help='Archivo de salida JSON'
    )

    args = parser.parse_args()

    # Crear descubridor
    discovery = SNMPDiscovery(
        seed_ips=args.seeds,
        community=args.community,
        timeout=args.timeout
    )

    # Descubrir
    devices = discovery.discover_recursive(max_depth=args.depth)

    # Resumen
    discovery.print_summary()

    # Exportar
    import json
    topology = discovery.export_to_json()

    with open(args.output, 'w') as f:
        json.dump(topology, f, indent=2)

    print(f"\n✓ Topología exportada a: {args.output}")
    print(f"\nPara visualizar: python3 visualize_topology.py --input {args.output}")


if __name__ == '__main__':
    main()
