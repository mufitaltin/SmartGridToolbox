#define BOOST_TEST_MODULE test_template
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE (test_template) // name of the test suite is stringtest

BOOST_AUTO_TEST_CASE (test1)
{
  BOOST_CHECK(0 == 0);
}

BOOST_AUTO_TEST_CASE (test2)
{
  BOOST_REQUIRE_EQUAL (1, 1); // basic test 
  BOOST_REQUIRE_EQUAL (1, 0); // basic test 
}

BOOST_AUTO_TEST_SUITE_END( )
