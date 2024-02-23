import datetime
import numpy as np
import math


def extract_codigo_nacional_assunto(dadosBasicos):
        if 'assunto' in dadosBasicos:
            assunto = dadosBasicos['assunto']
            assunto_codigos = []

            for a in assunto:
                if isinstance(a, dict):
                    if 'codigoNacional' in a:
                        codigo = a['codigoNacional']
                    elif 'assuntoLocal' in a:
                        codigo = a['assuntoLocal']['codigoPaiNacional']
                    elif 'codigoPaiNacional' in a:
                        codigo = a['codigoPaiNacional']
                    else:
                        codigo = None
                else:
                    codigo = None

                assunto_codigos.append(codigo)
            
            if not assunto_codigos:
                return None

            return assunto_codigos
        else:
            return None


def extract_classe_processual(dadosBasicos):
    temp = dadosBasicos['classeProcessual']
    
    if temp is None:
        return None

    if isinstance(temp, int) or isinstance(temp, float):
        return int(temp)
    
    if isinstance(temp, str) and temp.isdigit():
        return int(temp)
    
    return None


def extract_processo_eletronico(dadosBasicos):
    if 'procEl' in dadosBasicos:
        if type(dadosBasicos['procEl']) == int:
            return dadosBasicos['procEl']
        else:
            return -1
    

    return -1


def extract_nivel_sigilo(dadosBasicos):
    if 'nivelSigilo' in dadosBasicos:
        return dadosBasicos['nivelSigilo']
    

    return None


def extract_orgao_julgador_nome(dadosBasicos, siglaTribunal):
    if 'orgaoJulgador' in dadosBasicos:
        if 'nomeOrgao' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['nomeOrgao'] + \
            ' (' + siglaTribunal + ')'
    
    return None


def extract_orgao_julgador_codigo(dadosBasicos):
    if 'orgaoJulgador' in dadosBasicos:
        if 'codigoOrgao' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['codigoOrgao']

    return None


def extract_codigo_municipio_ibge(dadosBasicos):
    if 'orgaoJulgador' in dadosBasicos:
        if 'codigoMunicipioIBGE' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['codigoMunicipioIBGE']

    return -1


def extract_orgao_julgador_instancia(dadosBasicos):
    if 'orgaoJulgador' in dadosBasicos:
        if 'instancia' in dadosBasicos['orgaoJulgador']:
            return dadosBasicos['orgaoJulgador']['instancia']

    return None


def extract_numero_processo(dadosBasicos):

    if 'numero' in dadosBasicos:
        temp = dadosBasicos['numero']

    if temp == '':
        return None

    return temp


def extract_codigo_nacional_movimento(movimento):
    if movimento is not None and isinstance(movimento, dict):
        if 'movimentoNacional' in movimento and movimento['movimentoNacional'] \
            and 'codigoNacional' in movimento['movimentoNacional']:
            codigo = movimento['movimentoNacional']['codigoNacional']
            if codigo == 0:
                print()
        elif 'movimentoLocal' in movimento and movimento['movimentoLocal'] \
            and 'codigoPaiNacional' in movimento['movimentoLocal']:
            codigo = movimento['movimentoLocal']['codigoPaiNacional']
            # if codigo == 0:
            #     print()
        else:
            raise Exception('movimento invalido: ' + str(movimento))
        if isinstance(codigo, int):
            return codigo
        elif isinstance(codigo, str) and codigo.isdigit():
            return int(codigo)
        else:
            return None
    else:
        return None


def extract_data_hora_movimento(movimento):
    if movimento is not None and isinstance(movimento, dict):
        if isinstance(movimento['dataHora'], str):
            temp = movimento['dataHora']
        elif isinstance(movimento['dataHora'], int):
            temp = str(movimento['dataHora'])
        else:
            return None
        
        try:
            my_date = datetime.datetime.strptime(temp, '%Y%m%d%H%M%S')
            if my_date.year > 1900 and my_date.year <= 2023:
                return datetime.datetime.strptime(temp, '%Y%m%d%H%M%S')
            else:
                return None
        except:
            return None 
    else:
        return None


def parse_data(df, court):

    print('parsing data...')

    df['id'] = np.arange(len(df))
    df['id'] = df['id'].astype(str) + '_' + court

    df['court'] = court

    df['assuntoCodigoNacional'] = df['dadosBasicos'].\
        apply(extract_codigo_nacional_assunto)

    df['classeProcessual'] = df['dadosBasicos'].\
        apply(extract_classe_processual)

    df['orgaoJulgadorCodigo'] = df['dadosBasicos'].\
        apply(extract_orgao_julgador_codigo)
    
    df['codigoMunicipioIBGE'] = df['dadosBasicos'].\
        apply(extract_codigo_municipio_ibge)

    df['processoNumero'] = df['dadosBasicos'].\
        apply(extract_numero_processo)
    
    df['processoEletronico'] = df['dadosBasicos'].\
        apply(extract_processo_eletronico)
    
    df['nivelSigilo'] = df['dadosBasicos'].\
        apply(extract_nivel_sigilo)

    df = df.drop(columns=['dadosBasicos'])


    return df


def parse_data_mov(df_mov):
    df_mov['movimentoCodigoNacional'] = df_mov['movimento'].\
        apply(extract_codigo_nacional_movimento)
    df_mov['movimentoDataHora'] = df_mov['movimento'].\
        apply(extract_data_hora_movimento)

    df_mov = df_mov.drop(columns=['movimento',
                            'grau',
                            'siglaTribunal',
                            'millisInsercao',
                            ])


    return df_mov