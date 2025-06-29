trader\src\utils\trailing_stop.py:133: error: Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment]
trader\src\utils\gain_protector.py:83: note: By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked]
trader\src\exchange\symbol_cache.py:150: error: Unsupported target for indexed assignment ("object")  [index]
signal_aggregator\src\monitoring_stats.py:151: error: Function "builtins.any" is not valid as a type  [valid-type]
signal_aggregator\src\monitoring_stats.py:151: note: Perhaps you meant "typing.Any" instead of "any"?
signal_aggregator\src\monitoring_stats.py:193: error: Function "builtins.any" is not valid as a type  [valid-type]
signal_aggregator\src\monitoring_stats.py:193: note: Perhaps you meant "typing.Any" instead of "any"?
signal_aggregator\src\monitoring_stats.py:221: error: Need type annotation for "all_strategies" (hint: "all_strategies: set[<type>] = ...")  [var-annotated]
signal_aggregator\src\monitoring_stats.py:252: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
signal_aggregator\src\monitoring_stats.py:274: error: Function "builtins.any" is not valid as a type  [valid-type]
signal_aggregator\src\monitoring_stats.py:274: note: Perhaps you meant "typing.Any" instead of "any"?
signal_aggregator\src\monitoring_stats.py:304: error: Need type annotation for "regime_counts"  [var-annotated]
signal_aggregator\src\monitoring_stats.py:311: error: Argument "key" to "max" has incompatible type overloaded function; expected "Callable[[str], SupportsDunderLT[Any] | SupportsDunderGT[Any]]"  [arg-type]
signal_aggregator\src\monitoring_stats.py:315: error: Need type annotation for "all_strategies" (hint: "all_strategies: set[<type>] = ...")  [var-annotated]
signal_aggregator\src\monitoring_stats.py:329: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
coordinator\src\service_client.py:6: error: Library stubs not installed for "requests"  [import-untyped]
coordinator\src\service_client.py:54: error: No overload variant of "__sub__" of "datetime" matches argument type "None"  [operator]
coordinator\src\service_client.py:54: note: Possible overload variants:
coordinator\src\service_client.py:54: note:     def __sub__(self, datetime, /) -> timedelta
coordinator\src\service_client.py:54: note:     def __sub__(self, timedelta, /) -> datetime
coordinator\src\service_client.py:93: error: Need type annotation for "_cache" (hint: "_cache: dict[<type>, <type>] = ...")  [var-annotated]
coordinator\src\service_client.py:94: error: Need type annotation for "_cache_ttl" (hint: "_cache_ttl: dict[<type>, <type>] = ...")  [var-annotated]
coordinator\src\service_client.py:97: error: Incompatible default for argument "json_data" (default has type "None", argument has type "dict[Any, Any]")  [assignment]
coordinator\src\service_client.py:97: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
coordinator\src\service_client.py:97: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
coordinator\src\service_client.py:97: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[Any, Any]")  [assignment]
coordinator\src\service_client.py:169: error: Argument "params" to "_make_request" of "ServiceClient" has incompatible type "dict[str, str] | None"; expected "dict[Any, Any]"  [arg-type]
coordinator\src\service_client.py:204: error: Incompatible default for argument "metadata" (default has type "None", argument has type "dict[str, Any]")  [assignment]
coordinator\src\service_client.py:204: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
coordinator\src\service_client.py:204: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
coordinator\src\service_client.py:239: error: Incompatible default for argument "close_data" (default has type "None", argument has type "dict[str, Any]")  [assignment]
coordinator\src\service_client.py:239: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
coordinator\src\service_client.py:239: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
coordinator\src\cycle_sync_monitor.py:9: error: Library stubs not installed for "requests"  [import-untyped]
coordinator\src\cycle_sync_monitor.py:9: note: Hint: "python3 -m pip install types-requests"
coordinator\src\cycle_sync_monitor.py:9: note: (or run "mypy --install-types" to install all missing stub packages)
coordinator\src\cycle_sync_monitor.py:9: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports
coordinator\src\cycle_sync_monitor.py:35: error: Need type annotation for "known_cycles" (hint: "known_cycles: set[<type>] = ...")  [var-annotated]
coordinator\src\cycle_sync_monitor.py:202: error: Unsupported operand types for + ("None" and "int")  [operator]
coordinator\src\cycle_sync_monitor.py:202: note: Left operand is of type "int | None"
shared\src\config.py:116: error: "object" has no attribute "get"  [attr-defined]
shared\src\kafka_client.py:118: error: Need type annotation for "consumers" (hint: "consumers: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\kafka_client.py:119: error: Need type annotation for "consumer_threads" (hint: "consumer_threads: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\kafka_client.py:120: error: Need type annotation for "processor_threads" (hint: "processor_threads: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\kafka_client.py:121: error: Need type annotation for "message_queues" (hint: "message_queues: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\kafka_client.py:122: error: Need type annotation for "stop_events" (hint: "stop_events: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\kafka_client.py:123: error: Need type annotation for "topic_maps" (hint: "topic_maps: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\kafka_client.py:129: error: Need type annotation for "_existing_topics_cache" (hint: "_existing_topics_cache: set[<type>] = ...")  [var-annotated]
shared\src\kafka_client.py:271: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
shared\src\kafka_client.py:504: error: Need type annotation for "message_queue"  [var-annotated]
shared\src\db_pool.py:57: error: Incompatible default for argument "query_type" (default has type "None", argument has type "str")  [assignment]
shared\src\db_pool.py:57: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
shared\src\db_pool.py:57: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
shared\src\db_pool.py:57: error: Incompatible default for argument "query_text" (default has type "None", argument has type "str")  [assignment]
shared\src\db_pool.py:99: error: Incompatible default for argument "query_text" (default has type "None", argument has type "str")  [assignment]
shared\src\db_pool.py:99: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
shared\src\db_pool.py:99: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
shared\src\db_pool.py:219: error: Need type annotation for "available_connections"  [var-annotated]
shared\src\db_pool.py:222: error: Need type annotation for "in_use_connections" (hint: "in_use_connections: dict[<type>, <type>] = ...")  [var-annotated]
portfolio\src\binance_account_manager.py:10: error: Library stubs not installed for "requests"  [import-untyped]
portfolio\src\binance_account_manager.py:54: error: Need type annotation for "_prices_cache" (hint: "_prices_cache: dict[<type>, <type>] = ...")  [var-annotated]
portfolio\src\binance_account_manager.py:84: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
portfolio\src\binance_account_manager.py:84: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\binance_account_manager.py:84: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\binance_account_manager.py:276: error: "str" has no attribute "get"  [attr-defined]
portfolio\src\binance_account_manager.py:277: error: "str" has no attribute "get"  [attr-defined]
portfolio\src\binance_account_manager.py:282: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
portfolio\src\binance_account_manager.py:355: error: Incompatible return value type (got "dict[str, Any]", expected "list[dict[str, Any]]")  [return-value]
portfolio\src\binance_account_manager.py:386: error: Incompatible return value type (got "dict[str, Any]", expected "list[dict[str, Any]]")  [return-value]
signal_aggregator\src\signal_aggregator.py:33: error: Need type annotation for "data_history"  [var-annotated]
signal_aggregator\src\signal_aggregator.py:34: error: Need type annotation for "last_update"  [var-annotated]
signal_aggregator\src\signal_aggregator.py:57: error: Incompatible default for argument "limit" (default has type "None", argument has type "int")  [assignment]
signal_aggregator\src\signal_aggregator.py:57: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
signal_aggregator\src\signal_aggregator.py:57: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
signal_aggregator\src\signal_aggregator.py:238: error: "SignalAggregator" has no attribute "_apply_enhanced_regime_filtering"  [attr-defined]
signal_aggregator\src\signal_aggregator.py:353: error: Incompatible return value type (got "None", expected "dict[str, Any]")  [return-value]
signal_aggregator\src\signal_aggregator.py:364: error: Incompatible return value type (got "None", expected "dict[str, Any]")  [return-value]
signal_aggregator\src\signal_aggregator.py:372: error: Incompatible return value type (got "None", expected "dict[str, Any]")  [return-value]
signal_aggregator\src\signal_aggregator.py:435: error: Incompatible return value type (got "None", expected "dict[str, Any]")  [return-value]
signal_aggregator\src\signal_aggregator.py:584: error: "SignalAggregator" has no attribute "_apply_volume_boost"  [attr-defined]
signal_aggregator\src\signal_aggregator.py:587: error: "SignalAggregator" has no attribute "_apply_multi_strategy_bonus"  [attr-defined]
signal_aggregator\src\signal_aggregator.py:802: error: "SignalAggregator" has no attribute "_apply_volume_boost"  [attr-defined]
signal_aggregator\src\signal_aggregator.py:805: error: "SignalAggregator" has no attribute "_apply_multi_strategy_bonus"  [attr-defined]
signal_aggregator\src\signal_aggregator.py:863: error: "SignalAggregator" has no attribute "_extract_volume_summary"  [attr-defined]
signal_aggregator\src\signal_aggregator.py:1253: error: Need type annotation for "price_targets" (hint: "price_targets: list[<type>] = ...")  [var-annotated]
signal_aggregator\src\signal_aggregator.py:1760: error: Dict entry 0 has incompatible type "str": "float"; expected "str": "int"  [dict-item]
signal_aggregator\src\signal_aggregator.py:1761: error: Dict entry 1 has incompatible type "str": "float"; expected "str": "int"  [dict-item]
signal_aggregator\src\signal_aggregator.py:1762: error: Dict entry 2 has incompatible type "str": "float"; expected "str": "int"  [dict-item]
signal_aggregator\src\signal_aggregator.py:1767: error: Dict entry 0 has incompatible type "str": "float"; expected "str": "int"  [dict-item]
signal_aggregator\src\signal_aggregator.py:1768: error: Dict entry 1 has incompatible type "str": "float"; expected "str": "int"  [dict-item]
signal_aggregator\src\regime_detector.py:113: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
signal_aggregator\src\regime_detector.py:117: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
signal_aggregator\src\dynamic_thresholds.py:196: error: Function "builtins.any" is not valid as a type  [valid-type]
signal_aggregator\src\dynamic_thresholds.py:196: note: Perhaps you meant "typing.Any" instead of "any"?
shared\src\redis_client.py:122: error: Need type annotation for "pubsub_connections" (hint: "pubsub_connections: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:123: error: Need type annotation for "pubsub_threads" (hint: "pubsub_threads: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:124: error: Need type annotation for "pubsub_callbacks" (hint: "pubsub_callbacks: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:125: error: Need type annotation for "pubsub_channels" (hint: "pubsub_channels: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:126: error: Need type annotation for "pubsub_stop_events" (hint: "pubsub_stop_events: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:129: error: Need type annotation for "message_queues" (hint: "message_queues: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:130: error: Need type annotation for "processor_threads" (hint: "processor_threads: dict[<type>, <type>] = ...")  [var-annotated]
shared\src\redis_client.py:156: error: Argument 1 to "ConnectionPool" has incompatible type "**dict[str, object]"; expected "CacheFactoryInterface | None"  [arg-type]
shared\src\redis_client.py:335: error: Need type annotation for "message_queue"  [var-annotated]
analyzer\src\indicators\indicator_cache.py:32: error: Need type annotation for "_cache" (hint: "_cache: dict[<type>, <type>] = ...")  [var-annotated]
analyzer\src\indicators\indicator_cache.py:33: error: Need type annotation for "_timestamps" (hint: "_timestamps: dict[<type>, <type>] = ...")  [var-annotated]
analyzer\src\indicators\indicator_cache.py:85: error: Argument "key" to "min" has incompatible type overloaded function; expected "Callable[[Any], SupportsDunderLT[Any] | SupportsDunderGT[Any]]"  [arg-type]
trader\src\utils\safety.py:21: error: List item 0 has incompatible type "type[Exception]"; expected "Exception"  [list-item]
trader\src\utils\safety.py:191: error: Incompatible default for argument "additional_info" (default has type "None", argument has type "dict[str, Any]")  [assignment]
trader\src\utils\safety.py:191: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
trader\src\utils\safety.py:191: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
trader\src\trading\price_monitor.py:143: error: Incompatible types in assignment (expression has type "Thread", variable has type "None")  [assignment]
trader\src\trading\price_monitor.py:148: error: "None" has no attribute "start"  [attr-defined]
signal_aggregator\src\main.py:15: error: Module "signal_aggregator" has no attribute "SignalAggregator"  [attr-defined]
signal_aggregator\src\main.py:15: error: Module "signal_aggregator" has no attribute "EnhancedSignalAggregator"  [attr-defined]
gateway\src\ultra_data_fetcher.py:174: error: Argument "params" to "get" of "ClientSession" has incompatible type "dict[str, object]"; expected "str | Mapping[str, str | SupportsInt | float | Sequence[str | SupportsInt | float]] | Sequence[tuple[str, str | SupportsInt | float | Sequence[str | SupportsInt | float]]] | None"  [arg-type]
gateway\src\ultra_data_fetcher.py:174: error: Argument "timeout" to "get" of "ClientSession" has incompatible type "int"; expected "ClientTimeout | _SENTINEL | None"  [arg-type]
gateway\src\ultra_data_fetcher.py:294: error: Argument "params" to "get" of "ClientSession" has incompatible type "dict[str, object]"; expected "str | Mapping[str, str | SupportsInt | float | Sequence[str | SupportsInt | float]] | Sequence[tuple[str, str | SupportsInt | float | Sequence[str | SupportsInt | float]]] | None"  [arg-type]
gateway\src\ultra_data_fetcher.py:294: error: Argument "timeout" to "get" of "ClientSession" has incompatible type "int"; expected "ClientTimeout | _SENTINEL | None"  [arg-type]
gateway\src\ultra_data_fetcher.py:315: error: Argument "timeout" to "get" of "ClientSession" has incompatible type "int"; expected "ClientTimeout | _SENTINEL | None"  [arg-type]
gateway\src\ultra_data_fetcher.py:370: error: Argument 1 to "append" of "list" has incompatible type "float"; expected "int"  [arg-type]
gateway\src\ultra_data_fetcher.py:502: error: Argument 1 to "append" of "list" has incompatible type "float"; expected "int"  [arg-type]
gateway\src\main.py:217: error: Cannot infer type of lambda  [misc]
dispatcher\src\message_router.py:24: error: Incompatible default for argument "redis_client" (default has type "None", argument has type "RedisClientPool")  [assignment]
dispatcher\src\message_router.py:24: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
dispatcher\src\message_router.py:24: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
dispatcher\src\message_router.py:55: error: Need type annotation for "enriched_data_cache" (hint: "enriched_data_cache: dict[<type>, <type>] = ...")  [var-annotated]
dispatcher\src\message_router.py:69: error: Need type annotation for "high_priority_queue"  [var-annotated]
dispatcher\src\message_router.py:72: error: Need type annotation for "message_queue"  [var-annotated]
dispatcher\src\message_router.py:107: error: Need type annotation for "_dedup_cache" (hint: "_dedup_cache: dict[<type>, <type>] = ...")  [var-annotated]
dispatcher\src\message_router.py:370: error: Incompatible default for argument "priority" (default has type "None", argument has type "str")  [assignment]
dispatcher\src\message_router.py:370: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
dispatcher\src\message_router.py:370: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
gateway\src\binance_ws.py:11: error: Module "websockets.exceptions" has no attribute "InvalidStatusCode"; maybe "InvalidStatus" or "InvalidState"?  [attr-defined]
gateway\src\binance_ws.py:33: error: Incompatible default for argument "symbols" (default has type "None", argument has type "list[str]")  [assignment]
gateway\src\binance_ws.py:33: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
gateway\src\binance_ws.py:33: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
gateway\src\binance_ws.py:33: error: Incompatible default for argument "kafka_client" (default has type "None", argument has type "KafkaClientPool")  [assignment]
gateway\src\binance_ws.py:69: error: Need type annotation for "ticker_cache" (hint: "ticker_cache: dict[<type>, <type>] = ...")  [var-annotated]
gateway\src\binance_ws.py:70: error: Need type annotation for "orderbook_cache" (hint: "orderbook_cache: dict[<type>, <type>] = ...")  [var-annotated]
gateway\src\binance_ws.py:73: error: Need type annotation for "price_buffers" (hint: "price_buffers: dict[<type>, <type>] = ...")  [var-annotated]
gateway\src\binance_ws.py:74: error: Need type annotation for "volume_buffers" (hint: "volume_buffers: dict[<type>, <type>] = ...")  [var-annotated]
gateway\src\binance_ws.py:75: error: Need type annotation for "rsi_buffers" (hint: "rsi_buffers: dict[<type>, <type>] = ...")  [var-annotated]
gateway\src\binance_ws.py:76: error: Need type annotation for "macd_buffers" (hint: "macd_buffers: dict[<type>, <type>] = ...")  [var-annotated]
gateway\src\binance_ws.py:100: error: Incompatible types in assignment (expression has type "ClientConnection", variable has type "None")  [assignment]
gateway\src\binance_ws.py:109: error: "None" has no attribute "send"  [attr-defined]
gateway\src\binance_ws.py:113: error: "None" has no attribute "recv"  [attr-defined]
gateway\src\binance_ws.py:742: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
gateway\src\binance_ws.py:744: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
gateway\src\binance_ws.py:760: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
gateway\src\binance_ws.py:762: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
gateway\src\binance_ws.py:874: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
shared\src\schemas.py:41: error: Argument "default_factory" to "Field" has incompatible type "type[dict[_KT, _VT]]"; expected "Callable[[], Never] | Callable[[dict[str, Any]], Never]"  [arg-type]
shared\src\schemas.py:104: error: Argument "default_factory" to "Field" has incompatible type "type[dict[_KT, _VT]]"; expected "Callable[[], Never] | Callable[[dict[str, Any]], Never]"  [arg-type]
trader\src\trading\cycle_repository.py:59: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:59: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:78: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:78: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:101: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:101: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:106: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:106: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:111: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:111: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:117: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:117: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:127: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:127: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:136: error: Value of type variable "Self" of "execute" of "Cursor" cannot be "str"  [type-var]
trader\src\trading\cycle_repository.py:136: error: Too few arguments for "execute" of "Cursor"  [call-arg]
trader\src\trading\cycle_repository.py:216: error: Incompatible types in assignment (7 tuple items are incompatible; 4 items are omitted)  [assignment]
trader\src\trading\cycle_repository.py:216: note: Expression tuple item 3 has type "str"; "float" expected; 
trader\src\trading\cycle_repository.py:216: note: Expression tuple item 6 has type "float"; "float | None" expected; 
trader\src\trading\cycle_repository.py:216: note: Expression tuple item 7 has type "float | None"; "str | None" expected; 
trader\src\trading\cycle_repository.py:346: error: Incompatible types in assignment (6 tuple items are incompatible; 3 items are omitted)  [assignment]
trader\src\trading\cycle_repository.py:346: note: Expression tuple item 4 has type "str"; "Any | bool" expected; 
trader\src\trading\cycle_repository.py:346: note: Expression tuple item 7 has type "str | None"; "float | None" expected; 
trader\src\trading\cycle_repository.py:346: note: Expression tuple item 16 has type "float | None"; "datetime" expected; 
trader\src\exchange\binance_utils.py:11: error: Library stubs not installed for "requests"  [import-untyped]
trader\src\exchange\binance_utils.py:96: error: Incompatible types in assignment (expression has type "str", target has type "int")  [assignment]
trader\src\exchange\binance_utils.py:548: error: Incompatible types in assignment (expression has type "str", target has type "int")  [assignment]
trader\src\exchange\binance_utils.py:637: error: Incompatible types in assignment (expression has type "str", target has type "int")  [assignment]
trader\src\exchange\binance_utils.py:640: error: Incompatible types in assignment (expression has type "str", target has type "int")  [assignment]
trader\src\api\routes.py:12: error: Library stubs not installed for "requests"  [import-untyped]
portfolio\src\models.py:61: error: Incompatible default for argument "db_url" (default has type "None", argument has type "str")  [assignment]
portfolio\src\models.py:61: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\models.py:61: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\models.py:154: error: Incompatible default for argument "params" (default has type "None", argument has type "tuple[Any, ...]")  [assignment]
portfolio\src\models.py:154: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\models.py:154: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\models.py:189: error: Item "None" of "Any | None" has no attribute "cursor"  [union-attr]
portfolio\src\models.py:201: error: Item "None" of "Any | None" has no attribute "commit"  [union-attr]
portfolio\src\models.py:217: error: Item "None" of "Any | None" has no attribute "close"  [union-attr]
portfolio\src\models.py:241: error: Item "None" of "Any | None" has no attribute "rollback"  [union-attr]
portfolio\src\models.py:271: error: Item "None" of "Any | None" has no attribute "cursor"  [union-attr]
portfolio\src\models.py:276: error: Item "None" of "Any | None" has no attribute "commit"  [union-attr]
portfolio\src\models.py:283: error: Item "None" of "Any | None" has no attribute "rollback"  [union-attr]
portfolio\src\models.py:310: error: Item "None" of "Any | None" has no attribute "cursor"  [union-attr]
portfolio\src\models.py:314: error: Item "None" of "Any | None" has no attribute "commit"  [union-attr]
portfolio\src\models.py:321: error: Item "None" of "Any | None" has no attribute "rollback"  [union-attr]
portfolio\src\models.py:353: error: Need type annotation for "_cache" (hint: "_cache: dict[<type>, <type>] = ...")  [var-annotated]
portfolio\src\models.py:354: error: Need type annotation for "_locks" (hint: "_locks: dict[<type>, <type>] = ...")  [var-annotated]
portfolio\src\models.py:390: error: Incompatible default for argument "prefix" (default has type "None", argument has type "str")  [assignment]
portfolio\src\models.py:390: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\models.py:390: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\models.py:412: error: Incompatible default for argument "db_manager" (default has type "None", argument has type "DBManager")  [assignment]
portfolio\src\models.py:412: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\models.py:412: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\models.py:670: error: Argument 1 to "append" of "list" has incompatible type "datetime"; expected "str"  [arg-type]
portfolio\src\models.py:674: error: Argument 1 to "append" of "list" has incompatible type "datetime"; expected "str"  [arg-type]
portfolio\src\models.py:678: error: List item 0 has incompatible type "int"; expected "str"  [list-item]
portfolio\src\models.py:678: error: List item 1 has incompatible type "int"; expected "str"  [list-item]
portfolio\src\models.py:683: error: Incompatible return value type (got "list[dict[Any, Any]] | dict[Any, Any]", expected "list[dict[str, Any]]")  [return-value]
portfolio\src\models.py:731: error: Incompatible return value type (got "list[dict[Any, Any]] | dict[Any, Any]", expected "list[dict[str, Any]]")  [return-value]
portfolio\src\models.py:756: error: Incompatible return value type (got "list[dict[Any, Any]] | dict[Any, Any]", expected "list[dict[str, Any]]")  [return-value]
portfolio\src\models.py:781: error: Incompatible return value type (got "list[dict[Any, Any]] | dict[Any, Any]", expected "list[dict[str, Any]]")  [return-value]
portfolio\src\models.py:802: error: Item "list[dict[Any, Any]]" of "list[dict[Any, Any]] | dict[Any, Any]" has no attribute "get"  [union-attr]
coordinator\src\smart_cycle_manager.py:8: error: Library stubs not installed for "requests"  [import-untyped]
coordinator\src\smart_cycle_manager.py:142: error: Argument 1 to "get" of "dict" has incompatible type "SignalStrength | None"; expected "SignalStrength"  [arg-type]
coordinator\src\smart_cycle_manager.py:226: error: Argument 1 to "_calculate_allocation_percentage" of "SmartCycleManager" has incompatible type "SignalStrength | None"; expected "SignalStrength"  [arg-type]
coordinator\src\smart_cycle_manager.py:252: error: Item "None" of "SignalStrength | None" has no attribute "value"  [union-attr]
coordinator\src\smart_cycle_manager.py:494: error: Unsupported operand types for / ("float" and "None")  [operator]
coordinator\src\smart_cycle_manager.py:494: note: Right operand is of type "float | None"
coordinator\src\smart_cycle_manager.py:497: error: Unsupported operand types for * ("float" and "None")  [operator]
coordinator\src\smart_cycle_manager.py:497: note: Right operand is of type "float | None"
coordinator\src\smart_cycle_manager.py:679: error: Unsupported right operand type for in ("object")  [operator]
coordinator\src\smart_cycle_manager.py:680: error: Unsupported target for indexed assignment ("object")  [index]
coordinator\src\smart_cycle_manager.py:687: error: Value of type "object" is not indexable  [index]
coordinator\src\smart_cycle_manager.py:688: error: Value of type "object" is not indexable  [index]
coordinator\src\smart_cycle_manager.py:689: error: Value of type "object" is not indexable  [index]
coordinator\src\smart_cycle_manager.py:690: error: Value of type "object" is not indexable  [index]
coordinator\src\smart_cycle_manager.py:690: error: Unsupported target for indexed assignment ("object")  [index]
coordinator\src\smart_cycle_manager.py:691: error: Unsupported operand types for + ("object" and "float")  [operator]
coordinator\src\smart_cycle_manager.py:694: error: Unsupported operand types for < ("int" and "object")  [operator]
coordinator\src\smart_cycle_manager.py:695: error: "object" has no attribute "__iter__"; maybe "__dir__" or "__str__"? (not iterable)  [attr-defined]
coordinator\src\smart_cycle_manager.py:696: error: Value of type "object" is not indexable  [index]
coordinator\src\smart_cycle_manager.py:697: error: Value of type "object" is not indexable  [index]
coordinator\src\signal_processor.py:40: error: Argument 1 to "defaultdict" has incompatible type "Callable[[], deque[Never]]"; expected "Callable[[], list[tuple[StrategySignal, float]]] | None"  [arg-type]
coordinator\src\signal_processor.py:40: error: Incompatible return value type (got "deque[Never]", expected "list[tuple[StrategySignal, float]]")  [return-value]
coordinator\src\signal_processor.py:332: error: Argument 1 to "get" of "dict" has incompatible type "SignalStrength | None"; expected "SignalStrength"  [arg-type]
analyzer\strategies\strategy_upgrader.py:105: error: Item "None" of "dict[str, Any] | None" has no attribute "copy"  [union-attr]
analyzer\strategies\strategy_upgrader.py:109: error: Item "None" of "dict[str, Any] | None" has no attribute "update"  [union-attr]
analyzer\strategies\base_strategy.py:31: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\base_strategy.py:31: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\base_strategy.py:31: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\base_strategy.py:42: error: Need type annotation for "data_buffer"  [var-annotated]
analyzer\strategies\base_strategy.py:164: error: Incompatible default for argument "metadata" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\base_strategy.py:164: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\base_strategy.py:164: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\base_strategy.py:236: error: Incompatible default for argument "atr_percent" (default has type "None", argument has type "float")  [assignment]
analyzer\strategies\base_strategy.py:236: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\base_strategy.py:236: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\signal_scorer.py:82: error: Incompatible default for argument "market_data" (default has type "None", argument has type "dict[Any, Any]")  [assignment]
analyzer\src\signal_scorer.py:82: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\signal_scorer.py:82: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\signal_scorer.py:127: error: Unsupported operand types for * ("None" and "int")  [operator]
analyzer\src\signal_scorer.py:127: note: Left operand is of type "float | None"
analyzer\src\signal_scorer.py:197: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\src\signal_scorer.py:234: error: Incompatible default for argument "market_data" (default has type "None", argument has type "dict[Any, Any]")  [assignment]
analyzer\src\signal_scorer.py:234: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\signal_scorer.py:234: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\signal_scorer.py:340: error: Incompatible default for argument "market_data" (default has type "None", argument has type "dict[Any, Any]")  [assignment]
analyzer\src\signal_scorer.py:340: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\signal_scorer.py:340: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\signal_scorer.py:422: error: Incompatible default for argument "market_data" (default has type "None", argument has type "dict[Any, Any]")  [assignment]
analyzer\src\signal_scorer.py:422: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\signal_scorer.py:422: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\redis_subscriber.py:33: error: Incompatible default for argument "symbols" (default has type "None", argument has type "list[str]")  [assignment]
analyzer\src\redis_subscriber.py:33: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\redis_subscriber.py:33: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\redis_subscriber.py:52: error: Need type annotation for "data_cache" (hint: "data_cache: dict[<type>, <type>] = ...")  [var-annotated]
analyzer\src\redis_subscriber.py:68: error: Need type annotation for "market_data_queue"  [var-annotated]
analyzer\src\redis_subscriber.py:175: error: Incompatible types in assignment (expression has type "Thread", variable has type "None")  [assignment]
analyzer\src\redis_subscriber.py:180: error: "None" has no attribute "start"  [attr-defined]
analyzer\src\redis_subscriber.py:272: error: Missing positional argument "client_id" in call to "unsubscribe" of "RedisClientPool"  [call-arg]
trader\src\trading\stop_manager_pure.py:9: error: Library stubs not installed for "requests"  [import-untyped]
trader\src\trading\stop_manager_pure.py:179: error: Argument "entry_price" to "TrailingStop" has incompatible type "float | None"; expected "float"  [arg-type]
trader\src\trading\stop_manager_pure.py:204: error: Unsupported operand types for > ("float" and "None")  [operator]
trader\src\trading\stop_manager_pure.py:204: note: Right operand is of type "float | None"
trader\src\trading\stop_manager_pure.py:214: error: Unsupported operand types for < ("float" and "None")  [operator]
trader\src\trading\stop_manager_pure.py:214: note: Right operand is of type "float | None"
trader\src\trading\stop_manager_pure.py:235: error: Argument "entry_price" to "initialize_cycle" of "GainProtector" has incompatible type "float | None"; expected "float"  [arg-type]
trader\src\trading\stop_manager_pure.py:339: error: Value of type "object" is not indexable  [index]
trader\src\trading\stop_manager_pure.py:391: error: Incompatible types in assignment (expression has type "object", variable has type "dict[Any, Any]")  [assignment]
trader\src\trading\stop_manager_pure.py:393: error: Argument 1 has incompatible type "object"; expected "str"  [arg-type]
trader\src\trading\stop_manager_pure.py:395: error: Argument 3 has incompatible type "object"; expected "float"  [arg-type]
coordinator\src\signal_handler_refactored.py:72: error: Need type annotation for "signal_queue"  [var-annotated]
coordinator\src\signal_handler_refactored.py:606: error: Incompatible types in assignment (expression has type "Thread", variable has type "None")  [assignment]
coordinator\src\signal_handler_refactored.py:610: error: "None" has no attribute "daemon"  [attr-defined]
coordinator\src\signal_handler_refactored.py:611: error: "None" has no attribute "start"  [attr-defined]
coordinator\src\signal_handler_refactored.py:631: error: "RedisClientPool" has no attribute "unsubscribe_all"; maybe "unsubscribe"?  [attr-defined]
coordinator\src\signal_handler_refactored.py:694: error: Argument 1 to "remove_cycle_from_cache" of "CycleSyncMonitor" has incompatible type "Any | None"; expected "str"  [arg-type]
coordinator\src\signal_handler_refactored.py:710: error: Argument 1 to "remove_cycle_from_cache" of "CycleSyncMonitor" has incompatible type "Any | None"; expected "str"  [arg-type]
coordinator\src\signal_handler_refactored.py:726: error: Argument 1 to "remove_cycle_from_cache" of "CycleSyncMonitor" has incompatible type "Any | None"; expected "str"  [arg-type]
analyzer\strategies\ultra_confluence.py:32: error: Need type annotation for "mtf_data"  [var-annotated]
analyzer\strategies\ultra_confluence.py:86: error: Incompatible return value type (got "Any | bool | None", expected "bool")  [return-value]
analyzer\strategies\ultra_confluence.py:206: error: Argument "timestamp" to "StrategySignal" has incompatible type "float"; expected "datetime"  [arg-type]
analyzer\strategies\ultra_confluence.py:242: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:245: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:273: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:276: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:292: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:293: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:327: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:329: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:336: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:337: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:341: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:342: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:379: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\ultra_confluence.py:382: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
analyzer\strategies\rsi.py:32: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\rsi.py:32: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\rsi.py:32: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\rsi.py:226: error: Item "None" of "dict[str, Any] | None" has no attribute "update"  [union-attr]
analyzer\strategies\ride_or_react.py:36: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\ride_or_react.py:36: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\ride_or_react.py:36: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\ride_or_react.py:61: error: Need type annotation for "price_history" (hint: "price_history: dict[<type>, <type>] = ...")  [var-annotated]
analyzer\strategies\ride_or_react.py:62: error: Need type annotation for "timestamps" (hint: "timestamps: list[<type>] = ...")  [var-annotated]
analyzer\strategies\ride_or_react.py:69: error: Cannot assign to a method  [method-assign]
analyzer\strategies\ride_or_react.py:168: error: Signature of "calculate_atr" incompatible with supertype "BaseStrategy"  [override]
analyzer\strategies\ride_or_react.py:168: note:      Superclass:
analyzer\strategies\ride_or_react.py:168: note:          def calculate_atr(self, df: Any = ..., period: int = ...) -> float
analyzer\strategies\ride_or_react.py:168: note:      Subclass:
analyzer\strategies\ride_or_react.py:168: note:          def calculate_atr(self, period: int = ...) -> float
analyzer\strategies\ride_or_react.py:237: error: Incompatible types in assignment (expression has type "str", target has type "float")  [assignment]
analyzer\strategies\ride_or_react.py:238: error: Incompatible types in assignment (expression has type "str", target has type "float")  [assignment]
analyzer\strategies\ride_or_react.py:240: error: Incompatible types in assignment (expression has type "str", target has type "float")  [assignment]
analyzer\strategies\ride_or_react.py:241: error: Incompatible types in assignment (expression has type "str", target has type "float")  [assignment]
analyzer\strategies\ride_or_react.py:245: error: Incompatible types in assignment (expression has type "str", target has type "float")  [assignment]
analyzer\strategies\ride_or_react.py:276: error: Incompatible types in assignment (expression has type "dict[str, Any]", variable has type "None")  [assignment]
analyzer\strategies\ride_or_react.py:278: error: Incompatible types in assignment (expression has type "datetime", variable has type "None")  [assignment]
analyzer\strategies\reversal_divergence.py:34: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\reversal_divergence.py:34: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\reversal_divergence.py:34: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\reversal_divergence.py:57: error: Cannot assign to a method  [method-assign]
analyzer\strategies\reversal_divergence.py:313: error: Signature of "_validate_trend_alignment_for_signal" incompatible with supertype "BaseStrategy"  [override]
analyzer\strategies\reversal_divergence.py:313: note:      Superclass:
analyzer\strategies\reversal_divergence.py:313: note:          def _validate_trend_alignment_for_signal(self) -> str | None
analyzer\strategies\reversal_divergence.py:313: note:      Subclass:
analyzer\strategies\reversal_divergence.py:313: note:          def _validate_trend_alignment_for_signal(self, df: Any) -> str | None
analyzer\strategies\macd.py:34: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\macd.py:34: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\macd.py:34: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\macd.py:48: error: "object" has no attribute "get"  [attr-defined]
analyzer\strategies\macd.py:49: error: "object" has no attribute "get"  [attr-defined]
analyzer\strategies\macd.py:50: error: "object" has no attribute "get"  [attr-defined]
analyzer\strategies\macd.py:54: error: "object" has no attribute "get"  [attr-defined]
analyzer\strategies\macd.py:60: error: Need type annotation for "macd_line" (hint: "macd_line: list[<type>] = ...")  [var-annotated]
analyzer\strategies\macd.py:61: error: Need type annotation for "signal_line" (hint: "signal_line: list[<type>] = ...")  [var-annotated]
analyzer\strategies\macd.py:62: error: Need type annotation for "histogram" (hint: "histogram: list[<type>] = ...")  [var-annotated]
analyzer\strategies\macd.py:316: error: Item "None" of "dict[str, Any] | None" has no attribute "update"  [union-attr]
analyzer\strategies\macd.py:737: error: Incompatible types in assignment (expression has type "float", target has type "dict[str, Any] | int | str | None")  [assignment]
analyzer\strategies\macd.py:737: error: "list[Any]" has no attribute "iloc"  [attr-defined]
analyzer\strategies\macd.py:738: error: Incompatible types in assignment (expression has type "float", target has type "dict[str, Any] | int | str | None")  [assignment]
analyzer\strategies\macd.py:738: error: "list[Any]" has no attribute "iloc"  [attr-defined]
analyzer\strategies\macd.py:739: error: Incompatible types in assignment (expression has type "float", target has type "dict[str, Any] | int | str | None")  [assignment]
analyzer\strategies\macd.py:739: error: "list[Any]" has no attribute "iloc"  [attr-defined]
analyzer\strategies\ema_cross.py:33: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\ema_cross.py:33: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\ema_cross.py:33: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\ema_cross.py:234: error: Item "None" of "dict[str, Any] | None" has no attribute "update"  [union-attr]
analyzer\strategies\breakout.py:33: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\breakout.py:33: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\breakout.py:33: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\breakout.py:50: error: Need type annotation for "detected_ranges" (hint: "detected_ranges: list[<type>] = ...")  [var-annotated]
analyzer\strategies\breakout.py:313: error: Item "None" of "dict[str, Any] | None" has no attribute "update"  [union-attr]
analyzer\strategies\breakout.py:429: error: Signature of "_validate_trend_alignment_for_signal" incompatible with supertype "BaseStrategy"  [override]
analyzer\strategies\breakout.py:429: note:      Superclass:
analyzer\strategies\breakout.py:429: note:          def _validate_trend_alignment_for_signal(self) -> str | None
analyzer\strategies\breakout.py:429: note:      Subclass:
analyzer\strategies\breakout.py:429: note:          def _validate_trend_alignment_for_signal(self, df: Any) -> str | None
analyzer\strategies\breakout.py:469: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:475: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:476: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:480: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:481: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:482: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:483: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\breakout.py:484: error: Name "breakout" is not defined  [name-defined]
analyzer\strategies\bollinger.py:33: error: Incompatible default for argument "params" (default has type "None", argument has type "dict[str, Any]")  [assignment]
analyzer\strategies\bollinger.py:33: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\strategies\bollinger.py:33: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\strategies\bollinger.py:87: error: Argument "matype" to "BBANDS" has incompatible type "int"; expected "MA_Type"  [arg-type]
analyzer\strategies\bollinger.py:219: error: Item "None" of "dict[str, Any] | None" has no attribute "update"  [union-attr]
analyzer\src\strategy_loader.py:30: error: Incompatible default for argument "symbols" (default has type "None", argument has type "list[str]")  [assignment]
analyzer\src\strategy_loader.py:30: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\strategy_loader.py:30: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\strategy_loader.py:30: error: Incompatible default for argument "strategy_dir" (default has type "None", argument has type "str")  [assignment]
analyzer\src\strategy_loader.py:123: error: Argument 1 to "float" has incompatible type "float | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
trader\src\exchange\binance_executor.py:46: error: Need type annotation for "demo_trades" (hint: "demo_trades: dict[<type>, <type>] = ...")  [var-annotated]
coordinator\src\main.py:14: error: Library stubs not installed for "requests"  [import-untyped]
analyzer\src\concurrent_analyzer.py:47: error: Need type annotation for "all_signals" (hint: "all_signals: dict[<type>, <type>] = ...")  [var-annotated]
analyzer\src\concurrent_analyzer.py:56: error: Generator has incompatible item type "int"; expected "bool"  [misc]
analyzer\src\concurrent_analyzer.py:56: error: Argument 1 to "len" has incompatible type "list[Any] | BaseException"; expected "Sized"  [arg-type]
analyzer\src\concurrent_analyzer.py:59: error: Incompatible return value type (got "dict[str, list[Any] | BaseException]", expected "dict[str, list[Any]]")  [return-value]
analyzer\src\concurrent_analyzer.py:145: error: Item "BaseException" of "list[Any] | BaseException" has no attribute "__iter__" (not iterable)  [union-attr]
trader\src\utils\cleanup_orphan_orders.py:112: error: Need type annotation for "orders_by_symbol" (hint: "orders_by_symbol: dict[<type>, <type>] = ...")  [var-annotated]
trader\src\trading\reconciliation.py:167: error: Incompatible types in assignment (expression has type "float", target has type "int")  [assignment]
trader\src\trading\reconciliation.py:177: error: Incompatible types in assignment (expression has type "float", target has type "int")  [assignment]
trader\src\trading\reconciliation.py:213: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\reconciliation.py:249: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\reconciliation.py:337: error: Incompatible types in assignment (expression has type "None", target has type "int")  [assignment]
trader\src\trading\cycle_manager.py:34: error: Incompatible default for argument "db_url" (default has type "None", argument has type "str")  [assignment]
trader\src\trading\cycle_manager.py:34: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
trader\src\trading\cycle_manager.py:34: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
trader\src\trading\cycle_manager.py:34: error: Incompatible default for argument "binance_executor" (default has type "None", argument has type "BinanceExecutor")  [assignment]
trader\src\trading\cycle_manager.py:43: error: Missing positional arguments "api_key", "api_secret" in call to "BinanceExecutor"  [call-arg]
trader\src\trading\cycle_manager.py:487: error: Argument 1 to "int" has incompatible type "str | None"; expected "str | Buffer | SupportsInt | SupportsIndex | SupportsTrunc"  [arg-type]
trader\src\trading\cycle_manager.py:508: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:619: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:620: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:621: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:631: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:652: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:686: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:687: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:688: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:698: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:719: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:734: error: Missing named argument "leverage" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:758: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:779: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:803: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:806: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:808: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:811: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:831: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:888: error: Argument 1 to "float" has incompatible type "float | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
trader\src\trading\cycle_manager.py:889: error: Argument 1 to "float" has incompatible type "float | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
trader\src\trading\cycle_manager.py:896: error: Argument 1 to "float" has incompatible type "float | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
trader\src\trading\cycle_manager.py:897: error: Argument 1 to "float" has incompatible type "float | None"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type]
trader\src\trading\cycle_manager.py:933: error: Unsupported operand types for * ("float" and "None")  [operator]
trader\src\trading\cycle_manager.py:933: error: Unsupported operand types for * ("None" and "float")  [operator]
trader\src\trading\cycle_manager.py:933: error: Unsupported left operand type for * ("None")  [operator]
trader\src\trading\cycle_manager.py:933: note: Both left and right operands are unions
trader\src\trading\cycle_manager.py:934: error: Unsupported operand types for * ("float" and "None")  [operator]
trader\src\trading\cycle_manager.py:934: note: Right operand is of type "Any | float | None"
trader\src\trading\cycle_manager.py:958: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:959: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:1007: error: Unsupported operand types for + ("float" and "None")  [operator]
trader\src\trading\cycle_manager.py:1007: note: Right operand is of type "Any | float | None"
trader\src\trading\cycle_manager.py:1051: error: "CycleManager" has no attribute "create_cycle_from_signal"  [attr-defined]
trader\src\trading\cycle_manager.py:1127: error: Unsupported operand types for * ("None" and "float")  [operator]
trader\src\trading\cycle_manager.py:1127: note: Left operand is of type "float | None"
trader\src\trading\cycle_manager.py:1182: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:1204: error: Unsupported operand types for < ("float" and "None")  [operator]
trader\src\trading\cycle_manager.py:1204: note: Right operand is of type "float | None"
trader\src\trading\cycle_manager.py:1219: error: Unsupported operand types for * ("float" and "None")  [operator]
trader\src\trading\cycle_manager.py:1219: error: Unsupported operand types for * ("None" and "float")  [operator]
trader\src\trading\cycle_manager.py:1219: error: Unsupported left operand type for * ("None")  [operator]
trader\src\trading\cycle_manager.py:1219: note: Both left and right operands are unions
trader\src\trading\cycle_manager.py:1264: error: Item "None" of "dict[str, Any] | None" has no attribute "get"  [union-attr]
trader\src\trading\cycle_manager.py:1276: error: Missing named argument "leverage" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1352: error: Unsupported operand types for * ("float" and "None")  [operator]
trader\src\trading\cycle_manager.py:1352: error: Unsupported operand types for * ("None" and "float")  [operator]
trader\src\trading\cycle_manager.py:1352: error: Unsupported left operand type for * ("None")  [operator]
trader\src\trading\cycle_manager.py:1352: note: Both left and right operands are unions
trader\src\trading\cycle_manager.py:1435: error: Item "None" of "dict[str, Any] | None" has no attribute "get"  [union-attr]
trader\src\trading\cycle_manager.py:1444: error: Unexpected keyword argument "id" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1444: error: Unexpected keyword argument "cycle_id" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1444: error: Unexpected keyword argument "type" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1444: error: Unexpected keyword argument "status" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1444: error: Unexpected keyword argument "created_at" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1444: error: Unexpected keyword argument "updated_at" for "TradeOrder"  [call-arg]
trader\src\trading\cycle_manager.py:1469: error: Unsupported right operand type for in ("dict[str, Any] | None")  [operator]
trader\src\trading\cycle_manager.py:1470: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:1472: error: Value of type "dict[str, Any] | None" is not indexable  [index]
trader\src\trading\cycle_manager.py:1484: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:1560: error: Unsupported target for indexed assignment ("dict[str, Any] | None")  [index]
trader\src\trading\cycle_manager.py:1665: error: Incompatible default for argument "metadata" (default has type "None", argument has type "dict[str, Any]")  [assignment]
trader\src\trading\cycle_manager.py:1665: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
trader\src\trading\cycle_manager.py:1665: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
trader\src\trading\cycle_manager.py:1704: error: Unsupported operand types for + ("None" and "float")  [operator]
trader\src\trading\cycle_manager.py:1704: note: Left operand is of type "float | None"
analyzer\src\multiproc_manager.py:57: error: Incompatible default for argument "symbols" (default has type "None", argument has type "list[str]")  [assignment]
analyzer\src\multiproc_manager.py:57: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
analyzer\src\multiproc_manager.py:57: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
analyzer\src\multiproc_manager.py:57: error: Incompatible default for argument "max_workers" (default has type "None", argument has type "int")  [assignment]
analyzer\src\multiproc_manager.py:108: error: Incompatible types in assignment (expression has type "None", variable has type "ConcurrentAnalyzer")  [assignment]
analyzer\src\multiproc_manager.py:275: error: Module has no attribute "now"  [attr-defined]
trader\src\trading\order_manager.py:31: error: Incompatible default for argument "symbols" (default has type "None", argument has type "list[str]")  [assignment]
trader\src\trading\order_manager.py:31: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
trader\src\trading\order_manager.py:31: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
trader\src\trading\order_manager.py:56: error: Need type annotation for "paused_symbols" (hint: "paused_symbols: set[<type>] = ...")  [var-annotated]
trader\src\trading\order_manager.py:57: error: Need type annotation for "paused_strategies" (hint: "paused_strategies: set[<type>] = ...")  [var-annotated]
trader\src\trading\order_manager.py:62: error: Need type annotation for "last_prices" (hint: "last_prices: dict[<type>, <type>] = ...")  [var-annotated]
trader\src\trading\order_manager.py:150: error: Incompatible types in assignment (expression has type "float", variable has type "int")  [assignment]
portfolio\src\api.py:55: error: Item "list[dict[Any, Any]]" of "list[dict[Any, Any]] | dict[Any, Any]" has no attribute "get"  [union-attr]
portfolio\src\api.py:330: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:330: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:330: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:368: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:368: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:368: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:391: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:391: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:391: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:440: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:440: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:440: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:503: error: Argument 1 to "append" of "list" has incompatible type "datetime"; expected "str"  [arg-type]
portfolio\src\api.py:507: error: Argument 1 to "append" of "list" has incompatible type "datetime"; expected "str"  [arg-type]
portfolio\src\api.py:510: error: Item "list[dict[Any, Any]]" of "list[dict[Any, Any]] | dict[Any, Any]" has no attribute "get"  [union-attr]
portfolio\src\api.py:528: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:528: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:528: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:554: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:554: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:554: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:573: error: Incompatible default for argument "response" (default has type "None", argument has type "Response")  [assignment]
portfolio\src\api.py:573: note: PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True
portfolio\src\api.py:573: note: Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase
portfolio\src\api.py:619: error: Argument 1 to "update_balances" of "PortfolioModel" has incompatible type "list[AssetBalance]"; expected "list[AssetBalance | dict[Any, Any]]"  [arg-type]
portfolio\src\api.py:619: note: "List" is invariant -- see https://mypy.readthedocs.io/en/stable/common_issues.html#variance
portfolio\src\api.py:619: note: Consider using "Sequence" instead, which is covariant
portfolio\src\api.py:731: error: Argument 1 to "clear" of "SharedCache" has incompatible type "str | None"; expected "str"  [arg-type]
Found 388 errors in 52 files (checked 83 source files)
