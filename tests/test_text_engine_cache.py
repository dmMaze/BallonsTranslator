import gc
import unittest
import weakref

from ballontranslator.ui.text_engine.cache import KeyedLruCache


class _FactoryArgument:
    pass


class KeyedLruCacheTest(unittest.TestCase):
    def test_factory_arguments_are_not_retained(self):
        cache = KeyedLruCache(2)
        calls = []

        def build(argument):
            calls.append(True)
            return 'value'

        argument = _FactoryArgument()
        reference = weakref.ref(argument)
        self.assertEqual(
            cache.get_or_create('key', build, argument),
            'value',
        )
        del argument
        gc.collect()

        self.assertIsNone(reference())
        self.assertEqual(
            cache.get_or_create('key', build, _FactoryArgument()),
            'value',
        )
        self.assertEqual(len(calls), 1)

    def test_access_updates_lru_order_and_insertion_is_bounded(self):
        cache = KeyedLruCache(2)
        cache.get_or_create('first', str, 1)
        cache.get_or_create('second', str, 2)
        self.assertEqual(
            cache.get_or_create('first', self.fail),
            '1',
        )

        cache.get_or_create('third', str, 3)

        self.assertEqual(tuple(cache), ('first', 'third'))
        self.assertNotIn('second', cache)


if __name__ == '__main__':
    unittest.main()
