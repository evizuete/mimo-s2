#!/usr/bin/env python3
"""
Network Topology Discovery Tool
Descubre automáticamente la topología de una red LAN usando:
- LLDP (Link Layer Discovery Protocol)
- CDP (Cisco Discovery Protocol)
- ARP (Address Resolution Protocol)
- SNMP (Simple Network Management Protocol)
- Ping sweeps
"""

import subprocess
import re
import socket
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from collections import defaultdict
import ipaddress
import time

try:
    from scapy.all import ARP, Ether, srp, sniff, LLDP, CDPv2_HDR

    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠️  scapy no instalado. Instalar con: pip install scapy")

try:
    from pysnmp.hlapi import *

    SNMP_AVAILABLE = True
except ImportError:
    SNMP_AVAILABLE = False
    print("⚠️  pysnmp no instalado. Instalar con: pip install pysnmp")


@dataclass
class NetworkDevice:
    """Representa un dispositivo en la red"""
    ip: str
    mac: str = ""
    hostname: str = ""
    device_type: str = "unknown"  # switch, router, server, workstation, firewall
    manufacturer: str = ""
    model: str = ""
    ports: Dict[str, 'NetworkPort'] = field(default_factory=dict)
    neighbors: Set[str] = field(default_factory=set)
    snmp_enabled: bool = False
    services: List[str] = field(default_factory=list)
    os_info: str = ""

    def __hash__(self):
        return hash(self.ip)

    def __eq__(self, other):
        if isinstance(other, NetworkDevice):
            return self.ip == other.ip
        return False


@dataclass
class NetworkPort:
    """Representa un puerto de red"""
    port_id: str
    port_name: str = ""
    status: str = "unknown"  # up, down, admin_down
    speed: str = ""
    duplex: str = ""
    vlan: str = ""
    connected_to: Optional[str] = None  # IP del dispositivo conectado


@dataclass
class NetworkLink:
    """Representa un enlace entre dos dispositivos"""
    device1: str  # IP
    device2: str  # IP
    port1: str = ""
    port2: str = ""
    protocol: str = ""  # LLDP, CDP, ARP, inference


class NetworkDiscovery:
    """Motor principal de descubrimiento de topología"""

    def __init__(self, network_range: str = "192.168.1.0/24"):
        """
        Args:
            network_range: Rango de red en formato CIDR (ej: 192.168.1.0/24)
        """
        self.network_range = network_range
        self.devices: Dict[str, NetworkDevice] = {}
        self.links: List[NetworkLink] = []

    def discover_all(self, methods: List[str] = None):
        """
        Ejecuta todos los métodos de descubrimiento disponibles

        Args:
            methods: Lista de métodos a usar. Default: todos los disponibles
                    ['arp', 'lldp', 'cdp', 'snmp', 'ping']
        """
        print("=" * 80)
        print("DESCUBRIMIENTO DE TOPOLOGÍA DE RED")
        print("=" * 80)
        print(f"Rango de red: {self.network_range}\n")

        if methods is None:
            methods = ['arp', 'ping', 'lldp', 'snmp']

        # 1. Descubrimiento básico de hosts (ARP + Ping)
        if 'arp' in methods or 'ping' in methods:
            print("\n[1/4] Descubriendo hosts activos...")
            self.discover_hosts()

        # 2. Identificación de dispositivos (fingerprinting)
        print("\n[2/4] Identificando tipos de dispositivos...")
        self.identify_devices()

        # 3. Descubrimiento de vecinos (LLDP/CDP)
        if SCAPY_AVAILABLE and ('lldp' in methods or 'cdp' in methods):
            print("\n[3/4] Descubriendo vecinos (LLDP/CDP)...")
            self.discover_neighbors()

        # 4. Información detallada (SNMP)
        if SNMP_AVAILABLE and 'snmp' in methods:
            print("\n[4/4] Recolectando información SNMP...")
            self.discover_via_snmp()

        # Inferir enlaces no descubiertos
        print("\n[BONUS] Infiriendo enlaces adicionales...")
        self.infer_links()

        return self.devices, self.links

    def discover_hosts(self):
        """Descubre hosts activos usando ARP scan"""
        print(f"  Escaneando rango: {self.network_range}")

        if SCAPY_AVAILABLE:
            # Método 1: ARP scan con scapy (más confiable)
            try:
                arp_request = ARP(pdst=self.network_range)
                broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
                arp_request_broadcast = broadcast / arp_request

                answered, unanswered = srp(arp_request_broadcast, timeout=2, verbose=False)

                for sent, received in answered:
                    ip = received.psrc
                    mac = received.hwsrc

                    device = NetworkDevice(ip=ip, mac=mac)
                    self.devices[ip] = device

                    print(f"    ✓ {ip:15s} - {mac}")

                print(f"  Total dispositivos encontrados: {len(self.devices)}")

            except Exception as e:
                print(f"  ⚠️  Error en ARP scan: {e}")
        else:
            # Método 2: Fallback usando ping (menos información)
            print("  ⚠️  Scapy no disponible, usando ping sweep...")
            network = ipaddress.ip_network(self.network_range, strict=False)

            for ip in network.hosts():
                ip_str = str(ip)
                if self._ping_host(ip_str):
                    device = NetworkDevice(ip=ip_str)
                    self.devices[ip_str] = device
                    print(f"    ✓ {ip_str}")

    def _ping_host(self, ip: str, timeout: int = 1) -> bool:
        """Verifica si un host responde a ping"""
        try:
            # -c 1: un solo paquete, -W timeout: tiempo de espera
            result = subprocess.run(
                ['ping', '-c', '1', '-W', str(timeout), ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return result.returncode == 0
        except Exception:
            return False

    def identify_devices(self):
        """Identifica el tipo de dispositivo mediante fingerprinting"""
        for ip, device in self.devices.items():
            print(f"  Identificando {ip}...")

            # 1. Resolver hostname
            try:
                hostname = socket.gethostbyaddr(ip)[0]
                device.hostname = hostname
                print(f"    Hostname: {hostname}")
            except:
                pass

            # 2. Port scanning básico para identificar tipo
            open_ports = self._scan_common_ports(ip)
            device.services = open_ports

            # 3. Clasificar por puertos abiertos
            device.device_type = self._classify_device(open_ports, device.hostname)
            print(f"    Tipo: {device.device_type}")

            # 4. Identificar fabricante por MAC
            if device.mac:
                device.manufacturer = self._get_vendor_from_mac(device.mac)
                if device.manufacturer:
                    print(f"    Fabricante: {device.manufacturer}")

    def _scan_common_ports(self, ip: str, timeout: float = 0.5) -> List[str]:
        """Escanea puertos comunes para identificar servicios"""
        common_ports = {
            22: 'SSH',
            23: 'Telnet',
            80: 'HTTP',
            443: 'HTTPS',
            161: 'SNMP',
            3389: 'RDP',
            8080: 'HTTP-Alt',
        }

        open_services = []

        for port, service in common_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()

            if result == 0:
                open_services.append(f"{service}:{port}")

        return open_services

    def _classify_device(self, services: List[str], hostname: str = "") -> str:
        """Clasifica el tipo de dispositivo basado en puertos y hostname"""
        services_str = ' '.join(services).lower()
        hostname_lower = hostname.lower()

        # Patrones de switches/routers
        if any(x in hostname_lower for x in ['switch', 'sw-', 'router', 'rtr-', 'gw-']):
            if any(x in hostname_lower for x in ['core', 'dist', 'access']):
                return 'switch'
            elif any(x in hostname_lower for x in ['router', 'gw', 'gateway']):
                return 'router'

        # Firewall
        if any(x in hostname_lower for x in ['fw-', 'firewall', 'fortigate', 'palo', 'checkpoint']):
            return 'firewall'

        # Servidor (muchos servicios)
        if len(services) >= 3 or 'HTTP' in services_str or 'HTTPS' in services_str:
            if any(x in hostname_lower for x in ['srv', 'server', 'db', 'web', 'app']):
                return 'server'

        # Dispositivo de red (tiene SNMP)
        if 'SNMP:161' in services:
            return 'network_device'

        # Por defecto
        return 'workstation'

    def _get_vendor_from_mac(self, mac: str) -> str:
        """Obtiene el fabricante desde la MAC (OUI lookup)"""
        # Diccionario simplificado de OUIs comunes
        oui_db = {
            '00:50:56': 'VMware',
            '00:0C:29': 'VMware',
            '00:1B:21': 'Intel',
            '00:24:E8': 'Cisco',
            '00:1D:45': 'Cisco',
            '00:1E:BD': 'Cisco',
            'F0:9F:C2': 'Ubiquiti',
            'DC:9F:DB': 'Ubiquiti',
            '00:27:22': 'Fortinet',
            '70:85:C2': 'Fortinet',
            'B8:27:EB': 'Raspberry Pi',
            'DC:A6:32': 'Raspberry Pi',
        }

        oui = mac[:8].upper()
        return oui_db.get(oui, "")

    def discover_neighbors(self):
        """Descubre vecinos usando LLDP y CDP (requiere scapy y permisos root)"""
        if not SCAPY_AVAILABLE:
            print("  ⚠️  Scapy no disponible")
            return

        try:
            print("  Escuchando paquetes LLDP/CDP (esto puede tomar 30-60 segundos)...")
            print("  ⚠️  Requiere permisos de root/admin")

            # Capturar paquetes LLDP y CDP
            packets = sniff(
                filter="ether proto 0x88cc or ether dst 01:00:0c:cc:cc:cc",
                timeout=30,
                store=True
            )

            for packet in packets:
                if packet.haslayer(LLDP):
                    self._process_lldp_packet(packet)
                elif packet.haslayer(CDPv2_HDR):
                    self._process_cdp_packet(packet)

            print(f"  ✓ Enlaces descubiertos: {len(self.links)}")

        except PermissionError:
            print("  ❌ Error: Se requieren permisos de root para capturar paquetes")
            print("     Ejecutar con: sudo python3 script.py")
        except Exception as e:
            print(f"  ⚠️  Error capturando paquetes: {e}")

    def _process_lldp_packet(self, packet):
        """Procesa un paquete LLDP para extraer información de vecinos"""
        try:
            # Extraer información del paquete LLDP
            # (Implementación simplificada)
            src_mac = packet[Ether].src

            # Buscar dispositivo origen por MAC
            src_device = None
            for ip, dev in self.devices.items():
                if dev.mac == src_mac:
                    src_device = dev
                    break

            if src_device:
                # Aquí procesarías los TLVs de LLDP para extraer:
                # - System Name
                # - Port ID
                # - Neighbor info
                pass  # Implementación completa requiere parseo de TLVs

        except Exception as e:
            pass

    def _process_cdp_packet(self, packet):
        """Procesa un paquete CDP para extraer información de vecinos"""
        # Similar a LLDP
        pass

    def discover_via_snmp(self, community: str = "public"):
        """Descubre información detallada via SNMP"""
        if not SNMP_AVAILABLE:
            print("  ⚠️  pysnmp no disponible")
            return

        for ip, device in list(self.devices.items()):
            if 'SNMP:161' in device.services:
                print(f"  Consultando SNMP en {ip}...")

                try:
                    # OIDs comunes
                    oids = {
                        'sysName': '1.3.6.1.2.1.1.5.0',
                        'sysDescr': '1.3.6.1.2.1.1.1.0',
                        'sysUpTime': '1.3.6.1.2.1.1.3.0',
                    }

                    for name, oid in oids.items():
                        iterator = getCmd(
                            SnmpEngine(),
                            CommunityData(community),
                            UdpTransportTarget((ip, 161), timeout=2, retries=1),
                            ContextData(),
                            ObjectType(ObjectIdentity(oid))
                        )

                        errorIndication, errorStatus, errorIndex, varBinds = next(iterator)

                        if not errorIndication and not errorStatus:
                            for varBind in varBinds:
                                value = varBind[1].prettyPrint()
                                if name == 'sysName' and value:
                                    device.hostname = value
                                elif name == 'sysDescr' and value:
                                    device.os_info = value

                            device.snmp_enabled = True

                    print(f"    ✓ SNMP OK: {device.hostname}")

                except Exception as e:
                    print(f"    ⚠️  Error SNMP: {e}")

    def infer_links(self):
        """Infiere enlaces basándose en la topología conocida"""
        # Buscar gateway/router principal
        gateway_candidates = [
            dev for dev in self.devices.values()
            if dev.device_type in ['router', 'firewall', 'gateway']
        ]

        if gateway_candidates:
            gateway = gateway_candidates[0]
            print(f"  Gateway detectado: {gateway.ip}")

            # Asumir que todos los dispositivos se conectan al gateway
            for ip, device in self.devices.items():
                if ip != gateway.ip and device.device_type != 'router':
                    link = NetworkLink(
                        device1=gateway.ip,
                        device2=ip,
                        protocol="inference"
                    )
                    if link not in self.links:
                        self.links.append(link)

    def export_to_dict(self) -> Dict:
        """Exporta la topología a un diccionario"""
        return {
            'devices': {
                ip: {
                    'ip': dev.ip,
                    'mac': dev.mac,
                    'hostname': dev.hostname,
                    'type': dev.device_type,
                    'manufacturer': dev.manufacturer,
                    'services': dev.services,
                    'os': dev.os_info,
                }
                for ip, dev in self.devices.items()
            },
            'links': [
                {
                    'source': link.device1,
                    'target': link.device2,
                    'port1': link.port1,
                    'port2': link.port2,
                    'protocol': link.protocol,
                }
                for link in self.links
            ]
        }

    def print_summary(self):
        """Imprime un resumen de la topología descubierta"""
        print("\n" + "=" * 80)
        print("RESUMEN DE TOPOLOGÍA")
        print("=" * 80)

        # Contar por tipo
        type_count = defaultdict(int)
        for dev in self.devices.values():
            type_count[dev.device_type] += 1

        print(f"\nTotal dispositivos: {len(self.devices)}")
        print("\nPor tipo:")
        for dev_type, count in sorted(type_count.items()):
            print(f"  {dev_type:20s}: {count}")

        print(f"\nTotal enlaces: {len(self.links)}")

        print("\n" + "=" * 80)
        print("DISPOSITIVOS DETECTADOS")
        print("=" * 80)

        for ip, dev in sorted(self.devices.items()):
            print(f"\n{ip:15s} ({dev.device_type})")
            if dev.hostname:
                print(f"  Hostname: {dev.hostname}")
            if dev.mac:
                print(f"  MAC: {dev.mac}")
            if dev.manufacturer:
                print(f"  Fabricante: {dev.manufacturer}")
            if dev.services:
                print(f"  Servicios: {', '.join(dev.services)}")


def main():
    """Función principal"""
    import sys

    # Configuración
    network_range = "10.1.21.1/24"  # ⬅️ CAMBIAR SEGÚN TU RED

    if len(sys.argv) > 1:
        network_range = sys.argv[1]

    print(f"Iniciando descubrimiento en red: {network_range}")
    print(f"Nota: Algunos métodos requieren permisos de root\n")

    # Crear instancia y descubrir
    discovery = NetworkDiscovery(network_range)
    devices, links = discovery.discover_all()

    # Mostrar resumen
    discovery.print_summary()

    # Exportar a JSON
    import json
    topology = discovery.export_to_dict()

    with open('network_topology.json', 'w') as f:
        json.dump(topology, f, indent=2)

    print(f"\n✓ Topología exportada a: network_topology.json")
    print(f"\nPara visualizar gráficamente, usar: python3 visualize_topology.py")


if __name__ == '__main__':
    main()